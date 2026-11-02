#!/usr/bin/env python
"""Evaluate a MQ-MINER checkpoint: rerank a first-stage retrieval run and score it.

For each query, the retriever's top-`depth` pages are re-scored by the reranker and re-sorted;
ranks below `depth` keep the retriever's order. nDCG@k and Recall@k are averaged over the
queries in qrels, then over subsets, and reported next to the retriever alone (paper: depth 30,
nDCG@5 and Recall@5).

    python scripts/evaluate.py --data_dir data/vidore_v3 --runs runs/webai/vidore_v3 --depth 30 --out results/webai_vidore_v3
    torchrun --nproc_per_node=2 scripts/evaluate.py ...        # queries are split across GPUs

Inputs, per subset <s> (as written by data.py):
  <data_dir>/<s>/{queries.json, qrels.json, corpus_metadata.json, corpus_images/}
  <runs>/<s>.json   first-stage run: {query_id: [{"corpus_id": ..., "score": ...}, ...]}, best first
Outputs in --out: <s>.json (reranked run, same format) and metrics.json. With --by_language,
metrics are also split by the queries' `language` field (ViDoRe v3 has six).
"""
import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

from train import build_inputs, load_model


# =============================== metrics ===============================
def ndcg_at_k(ranked, qrel, k):
    dcg = lambda rels: sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))  # noqa: E731
    ideal = dcg(sorted(qrel.values(), reverse=True))
    return dcg([qrel.get(c, 0) for c in ranked]) / ideal if ideal > 0 else 0.0


def recall_at_k(ranked, qrel, k):
    rel = {c for c, s in qrel.items() if s > 0}
    return sum(c in rel for c in ranked[:k]) / len(rel) if rel else 0.0


def score_run(run, qrels, ks, qids=None):
    """Mean nDCG@k / Recall@k over the queries in qrels (restricted to qids); a query
    missing from the run scores 0."""
    qids = [q for q in qrels if qids is None or q in qids]
    out = {}
    for k in ks:
        out[f"ndcg@{k}"] = sum(ndcg_at_k(run.get(q, []), qrels[q], k) for q in qids) / len(qids)
        out[f"recall@{k}"] = sum(recall_at_k(run.get(q, []), qrels[q], k) for q in qids) / len(qids)
    return out


def mean(dicts):
    return {m: sum(d[m] for d in dicts) / len(dicts) for m in dicts[0]}


# =============================== reranking ===============================
@torch.no_grad()
def rerank(model, proc, dev, queries, img_path, run, qids, depth, batch_size):
    """{qid: [(corpus_id, reranker score)]} for the top-`depth` of each query in qids."""
    out = {}
    for n, q in enumerate(qids):
        cids = [c["corpus_id"] for c in run[q][:depth]]
        scores = []
        for i in range(0, len(cids), batch_size):
            chunk = cids[i:i + batch_size]
            inp = build_inputs(proc, [(queries[q], Image.open(img_path[c]).convert("RGB")) for c in chunk], dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scores += model(**inp).float().cpu().tolist()
        out[q] = list(zip(cids, scores))
        if (n + 1) % 100 == 0:
            print(f"  [{dev}] {n + 1}/{len(qids)} queries", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(Path(__file__).resolve().parents[1] / "weights" / "mq-miner.safetensors"),
                    help="released weights (default), a train.py checkpoint, or a Hub repo id")
    ap.add_argument("--data_dir", required=True, help="benchmark root, e.g. data/vidore_v3")
    ap.add_argument("--runs", required=True, help="dir with one first-stage run <subset>.json per subset")
    ap.add_argument("--subsets", nargs="+", help="default: every <subset>.json in --runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=30, help="rerank this many top candidates")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--by_language", action="store_true")
    ap.add_argument("--model_dir", default="models/Eagle2-1B")
    ap.add_argument("--max_tiles", type=int, default=6, help="6 in the paper (training used 4)")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dev, rank, world = f"cuda:{os.environ['LOCAL_RANK']}", dist.get_rank(), dist.get_world_size()
    else:
        dev, rank, world = "cuda", 0, 1
    is_main = rank == 0
    if ddp:   # one rank patches/downloads the base model
        if is_main:
            from train import patch_eagle2
            patch_eagle2(args.model_dir)
        dist.barrier()
    proc, model = load_model(args.model_dir, args.max_tiles, ckpt=args.ckpt)
    model.to(dev).eval()

    runs_dir, data_dir, out_dir = Path(args.runs), Path(args.data_dir), Path(args.out)
    subsets = args.subsets or sorted(p.stem for p in runs_dir.glob("*.json"))
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"checkpoint {args.ckpt}  depth {args.depth}  subsets {subsets}  (world={world})", flush=True)

    per_subset, per_lang = {}, defaultdict(lambda: defaultdict(list))
    for s in subsets:
        d = data_dir / s
        queries = {str(q["query_id"]): q for q in json.load(open(d / "queries.json"))}
        img_path = {c["corpus_id"]: d / c["image_path"] for c in json.load(open(d / "corpus_metadata.json"))}
        qrels = defaultdict(dict)
        for e in json.load(open(d / "qrels.json")):
            qrels[str(e["query_id"])][e["corpus_id"]] = e["score"]
        run = {str(q): v for q, v in json.load(open(runs_dir / f"{s}.json")).items()}

        qids = sorted(run)
        part = rerank(model, proc, dev, {q: queries[q]["query"] for q in qids}, img_path, run,
                      qids[rank::world], args.depth, args.batch_size)
        if ddp:
            parts = [None] * world
            dist.all_gather_object(parts, part)
            part = {q: v for p in parts for q, v in p.items()}
        if not is_main:
            continue

        reranked, base = {}, {}
        for q in qids:
            head = sorted(part[q], key=lambda x: -x[1])
            tail = [c["corpus_id"] for c in run[q][args.depth:]]
            reranked[q] = [c for c, _ in head] + tail
            base[q] = [c["corpus_id"] for c in run[q]]
        json.dump({q: [{"corpus_id": c, "score": sc} for c, sc in sorted(part[q], key=lambda x: -x[1])]
                   for q in qids}, open(out_dir / f"{s}.json", "w"))
        per_subset[s] = {"rerank": score_run(reranked, qrels, args.ks),
                         "retriever": score_run(base, qrels, args.ks)}
        if args.by_language:
            by = defaultdict(set)
            for q in qrels:
                by[queries[q].get("language", "unknown")].add(q)
            for lang, ids in by.items():
                per_lang[lang]["rerank"].append(score_run(reranked, qrels, args.ks, ids))
                per_lang[lang]["retriever"].append(score_run(base, qrels, args.ks, ids))
        m = per_subset[s]
        print(f"[{s}] " + "  ".join(f"{k} {m['rerank'][k]:.3f} ({m['retriever'][k]:.3f})"
                                    for k in m["rerank"] if k.endswith(("@5", "@10"))), flush=True)

    if is_main:
        res = {"ckpt": str(args.ckpt), "depth": args.depth, "subsets": per_subset,
               "mean": {w: mean([per_subset[s][w] for s in subsets]) for w in ("rerank", "retriever")}}
        if args.by_language:   # per language: mean over subsets
            res["by_language"] = {lang: {w: mean(v[w]) for w in v} for lang, v in per_lang.items()}
        json.dump(res, open(out_dir / "metrics.json", "w"), indent=1)
        print("mean  " + "  ".join(f"{k} {res['mean']['rerank'][k]:.3f} (retriever {res['mean']['retriever'][k]:.3f})"
                                   for k in res["mean"]["rerank"]), flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
