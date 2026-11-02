#!/usr/bin/env python
"""Train the MQ-MINER reranker: Eagle2-1B with LoRA on its language model + a linear head.

The page image and the prompt "Query: {query}\\nDoes this document page answer the query?" go
through Eagle2-1B; the last hidden states are mean-pooled over all tokens and a Linear(896->1)
gives the relevance score. Each training group is one query with its positive page(s) and hard
negatives, scored together; the default loss is listwise InfoNCE over the group (pointwise BCE
is available for ablations).

Paper recipe (defaults below; 2 GPUs x grad_accum 8 = 16 groups per optimizer step, 1 epoch,
the released checkpoint is step 9500):
    torchrun --nproc_per_node=2 scripts/train.py --train_set data/train_set.json --data_dir data \\
        --output_dir checkpoints/mq-miner
Single GPU: `python scripts/train.py ...` with --grad_accum 16 for the same global batch.
Resume after a crash or time limit: add --resume auto (restores optimizer, schedule and position).
Export a checkpoint for release (LoRA + head only, safetensors):
    python scripts/train.py export checkpoints/mq-miner/step9500.pt weights/mq-miner.safetensors

The base model is downloaded to --model_dir if missing and patched to use SDPA attention
instead of flash-attn (two edits; see patch_eagle2). Needs transformers 4.51 (requirements.txt).
"""
import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

# Eagle2's remote code calls torch.distributed.get_rank() even when not distributed
# (only to gate its own logging; 1 keeps it quiet, as in our runs).
_get_rank = dist.get_rank
dist.get_rank = lambda *a, **k: (_get_rank(*a, **k) if dist.is_initialized() else 1)

from peft import LoraConfig, inject_adapter_in_model  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModel, AutoProcessor  # noqa: E402

BASE_MODEL = "nvidia/Eagle2-1B"
PROMPT = "Query: {query}\nDoes this document page answer the query?"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# =============================== model ===============================
def patch_eagle2(model_dir):
    """Download Eagle2-1B if needed and make it run with SDPA instead of flash-attn: every
    `_attn_implementation` in config.json -> "sdpa", and stop the remote code from forcing
    flash_attention_2 on the SigLIP vision tower. Idempotent."""
    d = Path(model_dir)
    if not (d / "config.json").exists():
        from huggingface_hub import snapshot_download
        snapshot_download(BASE_MODEL, local_dir=d)
    cfg = json.load(open(d / "config.json"))
    if any(c.get("_attn_implementation") != "sdpa"
           for c in (cfg, cfg.get("vision_config", {}), cfg.get("text_config", {}))):
        for c in (cfg, cfg.get("vision_config", {}), cfg.get("text_config", {})):
            c["_attn_implementation"] = "sdpa"
        json.dump(cfg, open(d / "config.json", "w"), indent=2)
    p = d / "modeling_eagle2_5_vl.py"
    src = p.read_text()
    new = re.sub(r"(config\.vision_config\._attn_implementation = )'flash_attention_2'", r"\1'sdpa'", src)
    if new != src:
        p.write_text(new)
        # transformers caches remote code per model dir name; drop the stale unpatched copy
        cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        shutil.rmtree(cache / "modules/transformers_modules" / d.name, ignore_errors=True)


class Reranker(nn.Module):
    """Eagle2 (LoRA in the LLM) -> mean-pooled last hidden state -> Linear -> scalar score."""

    def __init__(self, model, head):
        super().__init__()
        self.model = model
        self.head = head
        self.image_token_index = getattr(model, "image_token_index", model.config.image_token_index)

    def forward(self, input_ids, attention_mask, pixel_values):
        m = self.model
        inp = m.language_model.get_input_embeddings()(input_ids)
        vit = m.extract_feature(pixel_values)                     # (tiles, 256, C)
        B, N, C = inp.shape
        inp = inp.reshape(B * N, C).clone()                       # clone -> grad-safe index write
        inp[input_ids.reshape(B * N) == self.image_token_index] = vit.reshape(-1, C).to(inp.dtype)
        inp = inp.reshape(B, N, C)
        pos = attention_mask.long().cumsum(-1) - 1                # left padding
        pos.masked_fill_(attention_mask == 0, 1)
        out = m.language_model.model(inputs_embeds=inp, attention_mask=attention_mask,
                                     position_ids=pos, use_cache=False, return_dict=True)
        last = out.last_hidden_state
        am = attention_mask.unsqueeze(-1).to(last.dtype)
        pooled = (last * am).sum(1) / am.sum(1).clamp(min=1)      # (B, C)
        return self.head(pooled.float()).squeeze(-1)              # (B,)


RELEASE_FILE = "mq-miner.safetensors"
LORA_KEYS = ("r", "lora_alpha", "lora_dropout", "target_modules", "bias")


def export_weights(ckpt, out):
    """Training checkpoint (.pt, with optimizer state) -> release file: LoRA + head tensors in
    safetensors, LoRA config in its metadata. Loads without unpickling."""
    from safetensors.torch import save_file
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = {k: ck["lora_config"][k] for k in LORA_KEYS}
    cfg["target_modules"] = sorted(cfg["target_modules"])
    tensors = {f"lora.{n}": t.contiguous() for n, t in ck["lora"].items()}
    tensors |= {f"head.{n}": t.contiguous() for n, t in ck["head"].items()}
    save_file(tensors, out, metadata={"lora_config": json.dumps(cfg), "base_model": BASE_MODEL,
                                      "global_step": str(ck["global_step"])})
    print(f"wrote {out}: {len(ck['lora'])} LoRA tensors + head, step {ck['global_step']}")


def read_weights(ckpt):
    """-> (lora state dict, head state dict, LoraConfig) from a release file (local path or a
    Hub model repo id holding mq-miner.safetensors) or a training checkpoint (.pt)."""
    if not Path(ckpt).exists() and str(ckpt).count("/") == 1:
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(ckpt, RELEASE_FILE)
    if str(ckpt).endswith(".safetensors"):
        from safetensors import safe_open
        with safe_open(ckpt, "pt") as f:
            cfg = json.loads(f.metadata()["lora_config"])
            t = {k: f.get_tensor(k) for k in f.keys()}
        lora = {k[5:]: v for k, v in t.items() if k.startswith("lora.")}
        head = {k[5:]: v for k, v in t.items() if k.startswith("head.")}
    else:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        lora, head, cfg = ck["lora"], ck["head"], {k: ck["lora_config"][k] for k in LORA_KEYS}
    return lora, head, LoraConfig(**cfg)


def load_model(model_dir, max_tiles=4, ckpt=None, lora_r=16, lora_alpha=32, lora_dropout=0.05):
    """-> (processor, Reranker). With ckpt (see read_weights), loads its LoRA + head weights."""
    patch_eagle2(model_dir)
    proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, use_fast=True)
    proc.tokenizer.padding_side = "left"
    for attr in ("max_dynamic_tiles", "max_num_tiles", "max_tiles"):
        if hasattr(proc.image_processor, attr):
            setattr(proc.image_processor, attr, max_tiles)
    base = AutoModel.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)
    ck = None
    if ckpt:
        lora, head_sd, lora_cfg = read_weights(ckpt)
        ck = {"lora": lora, "head": head_sd}
    else:
        lora_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                              target_modules=LORA_TARGETS, bias="none")
    inject_adapter_in_model(lora_cfg, base.language_model)
    base.config.use_cache = False
    head = nn.Linear(base.config.text_config.hidden_size, 1)
    if ck:
        res = base.load_state_dict(ck["lora"], strict=False)
        assert not res.unexpected_keys, f"unexpected LoRA keys: {res.unexpected_keys[:5]}"
        head.load_state_dict(ck["head"])
    model = Reranker(base, head)
    model.lora_cfg = lora_cfg
    return proc, model


def build_inputs(proc, examples, device, prompt=PROMPT):
    """examples: [(query_text, PIL image)] -> model inputs on device."""
    texts, images = [], []
    for query, img in examples:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt.format(query=query)}]}]
        texts.append(proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        images.extend(proc.process_vision_info(messages)[0])
    b = proc(text=texts, images=images, return_tensors="pt", padding=True)
    return {"input_ids": b["input_ids"].to(device),
            "attention_mask": b["attention_mask"].to(device),
            "pixel_values": b["pixel_values"].to(device).to(torch.bfloat16)}


# =============================== data ===============================
def training_group(cands, max_pos, group_size):
    """Positives (up to max_pos, retrieved ones first; else the fallback gold page) + the
    top-ranked hard negatives, group_size candidates in total."""
    pos = [c for c in cands if c["label"] == 1 and c["in_top_k"]]
    if pos:
        pos = sorted(pos, key=lambda c: (-c["qrels_score"], c["rank"]))[:max_pos]
    else:
        pos = sorted([c for c in cands if c["label"] == 1 and c["is_fallback"]],
                     key=lambda c: -c["qrels_score"])[:max_pos]
    negs = sorted([c for c in cands if c["label"] == 0 and c["in_top_k"]],
                  key=lambda c: c["rank"])[: max(0, group_size - len(pos))]
    return pos, negs


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "export":   # python scripts/train.py export <ckpt.pt> <out.safetensors>
        ap = argparse.ArgumentParser(prog="train.py export", description=export_weights.__doc__)
        ap.add_argument("ckpt")
        ap.add_argument("out", nargs="?", default=RELEASE_FILE)
        a = ap.parse_args(sys.argv[2:])
        return export_weights(a.ckpt, a.out)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_set", required=True, help="data.py assemble output")
    ap.add_argument("--data_dir", default="data", help="root the train set's image paths are relative to")
    ap.add_argument("--model_dir", default="models/Eagle2-1B")
    ap.add_argument("--output_dir", default="checkpoints/mq-miner")
    ap.add_argument("--loss", choices=["infonce", "pointwise"], default="infonce")
    ap.add_argument("--max_pos", type=int, default=3)
    ap.add_argument("--group_size", type=int, default=20, help="max candidates per group")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=None,
                    help="stop after this optimizer step (the LR schedule still spans --epochs)")
    ap.add_argument("--lr_lora", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--grad_accum", type=int, default=8, help="groups per GPU per optimizer step")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--max_tiles", type=int, default=4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--grad_checkpoint", action="store_true", help="less memory, ~30%% slower")
    ap.add_argument("--img_cache_size", type=int, default=4096, help="decoded pages kept per rank (LRU)")
    ap.add_argument("--max_train_queries", type=int, default=None, help="subsample (debugging)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--ckpt_every", type=int, default=250)
    ap.add_argument("--resume", default=None, help="checkpoint path, or 'auto' for the latest in output_dir")
    args = ap.parse_args()

    # ---- DDP (torchrun) or single GPU ----
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        # long timeout: rank 0 writes checkpoints while the others wait at barriers
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dev, rank, world = f"cuda:{local_rank}", dist.get_rank(), dist.get_world_size()
    else:
        dev, rank, world = "cuda", 0, 1
    is_main = rank == 0

    def log(*a):
        if is_main:
            print(*a, flush=True)

    # ---- data ----
    data = json.load(open(args.train_set))["queries"]
    groups = []
    for q in data.values():
        if q["split"] != "train":
            continue
        pos, negs = training_group(q["candidates"], args.max_pos, args.group_size)
        if pos and negs:
            groups.append([(q["query"], c["image_path"], 1) for c in pos] +
                          [(q["query"], c["image_path"], 0) for c in negs])
    del data
    if args.max_train_queries:
        groups = groups[: args.max_train_queries]
    log(f"train groups: {len(groups)}  (world={world})")

    image_root = Path(args.data_dir)
    img_cache = OrderedDict()   # bounded: a decoded page is ~9 MB

    def load_img(path):
        img = img_cache.get(path)
        if img is None:
            img = Image.open(image_root / path).convert("RGB")
            if args.img_cache_size:
                img_cache[path] = img
                if len(img_cache) > args.img_cache_size:
                    img_cache.popitem(last=False)
        else:
            img_cache.move_to_end(path)
        return img

    # ---- model ----
    if ddp:   # one rank patches/downloads the base model, the rest wait
        if is_main:
            patch_eagle2(args.model_dir)
        dist.barrier()
    proc, model = load_model(args.model_dir, args.max_tiles, lora_r=args.lora_r,
                             lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
    base, head = model.model, model.head
    for n, p in base.named_parameters():
        p.requires_grad = "lora_" in n
    if args.grad_checkpoint:
        base.language_model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(dev).train()
    fwd = model
    if ddp:
        fwd = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                  find_unused_parameters=False, static_graph=True)
    log(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    opt = torch.optim.AdamW([
        {"params": [p for p in base.parameters() if p.requires_grad], "lr": args.lr_lora},
        {"params": head.parameters(), "lr": args.lr_head},
    ], weight_decay=args.weight_decay)
    per_rank = len(groups) // world          # equal shards keep DDP backward counts in step
    total_steps = math.ceil(per_rank / args.grad_accum) * args.epochs
    warmup = max(1, int(total_steps * args.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / warmup) * max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    bce = nn.BCEWithLogitsLoss()

    # ---- checkpoints ----
    odir = Path(args.output_dir)
    if is_main:
        odir.mkdir(parents=True, exist_ok=True)

    def save(name, ep, micro, step):
        if not is_main:
            return
        tmp = odir / f"{name}.tmp"   # write + rename: a crash mid-save never corrupts a checkpoint
        torch.save({"lora": {n: p.detach().cpu() for n, p in base.named_parameters() if "lora_" in n},
                    "lora_config": model.lora_cfg.to_dict(), "head": head.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "micro": micro, "global_step": step, "args": vars(args)}, tmp)
        tmp.replace(odir / f"{name}.pt")
        (odir / "last.json").write_text(json.dumps({"name": name}))

    g = torch.Generator().manual_seed(args.seed)
    step, start_ep, skip = 0, 1, 0
    ckpt = None
    if args.resume == "auto" and (odir / "last.json").exists():
        ckpt = odir / f"{json.loads((odir / 'last.json').read_text())['name']}.pt"
    elif args.resume and args.resume != "auto":
        ckpt = Path(args.resume)
    if ckpt:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        res = base.load_state_dict(ck["lora"], strict=False)
        assert not res.unexpected_keys, res.unexpected_keys[:5]
        head.load_state_dict(ck["head"])
        model.to(dev)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, start_ep, skip = ck["global_step"], ck["epoch"], ck["micro"]
        if skip >= per_rank:      # that epoch was finished
            start_ep, skip = start_ep + 1, 0
        for _ in range(start_ep - 1):   # replay the shuffle RNG past finished epochs
            torch.randperm(len(groups), generator=g)
        log(f"resumed from {ckpt.name}: epoch {start_ep}, skipping {skip} groups, step {step}")

    # ---- train ----
    t0, step0 = time.time(), step
    done = False
    for ep in range(start_ep, args.epochs + 1):
        order = torch.randperm(len(groups), generator=g).tolist()
        order = order[rank::world][:per_rank]
        micro = skip if ep == start_ep else 0
        order = order[micro:]
        opt.zero_grad()
        running, nl = 0.0, 0
        for gi in order:
            group = groups[gi]
            labels = torch.tensor([it[2] for it in group], dtype=torch.float, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scores = fwd(**build_inputs(proc, [(it[0], load_img(it[1])) for it in group], dev))
                if args.loss == "infonce":
                    loss = -torch.log_softmax(scores, dim=0)[labels.bool()].mean()
                else:
                    loss = bce(scores, labels)
            (loss / args.grad_accum).backward()
            running += loss.item()
            micro += 1
            nl += 1
            if micro % args.grad_accum == 0:
                if args.grad_clip:
                    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if step % args.log_every == 0:
                    rate = (step - step0) / max(time.time() - t0, 1e-9)
                    log(f"step {step}/{total_steps}  ep {ep}  loss {running / nl:.4f}  {rate:.3f} step/s  "
                        f"eta {(total_steps - step) / max(rate, 1e-9) / 3600:.2f}h")
                if args.ckpt_every and step % args.ckpt_every == 0:
                    save(f"step{step}", ep, micro, step)
                if args.max_steps and step >= args.max_steps:
                    done = True
                    break
        if done:
            break
        if micro % args.grad_accum:
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
        log(f"epoch {ep} done  avg loss {running / max(nl, 1):.4f}")
        save(f"epoch{ep}", ep, per_rank, step)
        if ddp:
            dist.barrier()

    save("final", ep, micro, step)
    log(f"done: step {step}, checkpoints in {odir}")
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
