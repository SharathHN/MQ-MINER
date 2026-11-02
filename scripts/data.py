#!/usr/bin/env python
"""MQ-MINER data: build the training set and the two evaluation benchmarks.

Training set (from the original public sources + the released manifest):
    python scripts/data.py download  --data_dir data          # 7 sources at pinned Hub revisions (~77 GB)
    python scripts/data.py prepare   --data_dir data          # page images + queries/qrels per source (~85 GB)
    python scripts/data.py assemble  --data_dir data --manifest manifest/train_manifest_en.jsonl.gz manifest/train_manifest_ml.jsonl.gz
                                                      # -> data/train_set.json (what train.py reads)
  Pass only manifest/train_manifest_en.jsonl.gz for the English-only set.
  `--only <source ...>` limits download/prepare to some sources. Every step is resumable.

Benchmarks (ViDoRe v3, 8 subsets; UniDoc-Bench, 8 domains, 200 English queries each):
    python scripts/data.py vidore_v3 --data_dir data          # -> data/vidore_v3/{subset}/
    python scripts/data.py unidoc    --data_dir data          # -> data/unidoc/{domain}/
  Each subset holds corpus_metadata.json, queries.json, qrels.json and corpus_images/.

The Hub is called with retries (HF_HUB_DOWNLOAD_TIMEOUT=60): flaky proxies are expected.
"""
import argparse, fnmatch, glob, gzip, hashlib, io, json, os, textwrap, time
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
import pyarrow.parquet as pq  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

QUERY_COLS = ["query", "rephrase_level_1", "rephrase_level_2", "rephrase_level_3"]

# Upstream training sources, pinned to the Hub revisions the release was built from.
SOURCES = {
    "colpali_train_set": ("vidore/colpali_train_set", "e13d3594064836f7fd69fad7e3d2b51065b335c7",
                          ["data/*.parquet"]),
    "visrag_in_domain": ("openbmb/VisRAG-Ret-Train-In-domain-data", "9c2ecd955e15151b57c6e4ae7a7bb13dc5ad09d2",
                         ["data/train-*.parquet"]),
    "docvqa": ("lmms-lab/DocVQA", "539088ef8a8ada01ac8e2e6d4e372586748a265e",
               ["DocVQA/train-*.parquet", "DocVQA/validation-*.parquet"]),
    "infovqa_train": ("vidore/infovqa_train", "fd098dfce64d2ac5f699338bbe724d3b38e9f6a3",
                      ["data/train-*.parquet"]),
    "tatqa": ("next-tat/TAT-QA", "c96247f5077eac447f63527fd3dcfdc58bb56d6a",
              ["tatqa_dataset_train.json", "tatqa_dataset_dev.json"]),
    "realmm_finreport": ("ibm-research/REAL-MM-RAG_FinReport", "904a87dcaaf855a1389a6a93d98fb7efed74086c",
                         ["data/*.parquet"]),
    "realmm_finslides": ("ibm-research/REAL-MM-RAG_FinSlides", "a4fafbf2868cddcc79c4ee522e59b8f8b993cdde",
                         ["data/*.parquet"]),
}

# Benchmarks, pinned to the revisions we evaluated on.
VIDORE_V3 = {
    "finance_en": "7f432c176d82e27546501ad8064a713ac3071809",
    "pharmaceuticals": "3abd4aa8a9445fb5538a78a19ba50bd57bd22b5c",
    "computer_science": "d5cc75883d92e294f0c0fc2662551c9708a06ebc",
    "hr": "0cdf0979f2c5a0fd3e335e6373b9da48a9fe3bc3",
    "industrial": "e26c864724f5dd71a3d7d739272d95637764cee9",
    "energy": "caec06d3c73434d635f710f93bcd898331c59f20",
    "physics": "a0de276f515acc044b72cae8de53a44bb5a8f1f5",
    "finance_fr": "1d808daa08032ffecdf62da151a7f7a8fe2bd0c9",
}
UNIDOC = ("Salesforce/UniDoc-Bench", "2e22c592438d1e52c39bd769aab2901d596af746")
UNIDOC_DOMAINS = ["commerce_manufacturing", "construction", "crm", "education",
                  "energy", "finance", "healthcare", "legal"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def hub_download(repo, revision, patterns, dest, tries=60, workers=4):
    """snapshot_download until every file matching `patterns` at `revision` is on disk."""
    from huggingface_hub import HfApi, snapshot_download
    for a in range(1, tries + 1):
        try:
            files = HfApi().list_repo_files(repo, repo_type="dataset", revision=revision)
            break
        except Exception as e:  # noqa: BLE001
            log(f"{repo} list: attempt {a}/{tries}: {type(e).__name__}: {str(e)[:150]}")
            if a == tries:
                raise
            time.sleep(min(30, 3 * a))
    want = sorted(f for f in files if any(fnmatch.fnmatch(f, p) for p in patterns))
    log(f"=== {repo}@{revision[:7]} -> {dest} ({len(want)} files)")
    for a in range(1, tries + 1):
        if all((dest / f).exists() for f in want):
            break
        try:
            snapshot_download(repo_id=repo, repo_type="dataset", revision=revision, local_dir=dest,
                              allow_patterns=patterns, max_workers=workers)
        except Exception as e:  # noqa: BLE001
            log(f"{repo}: attempt {a}/{tries}: {type(e).__name__}: {str(e)[:150]}")
            time.sleep(min(30, 3 * a))
    missing = [f for f in want if not (dest / f).exists()]
    if missing:
        raise SystemExit(f"{repo}: still missing {len(missing)} files, e.g. {missing[:3]}")


# =============================== training set: download ===============================
def download(root, only):
    for name, (repo, rev, patterns) in SOURCES.items():
        if not only or name in only:
            hub_download(repo, rev, patterns, root / "raw" / name)


# =============================== training set: prepare ===============================
# Converts raw/<source> into prepared/<subset>/. The manifest's corpus_id / orig_qid refer to
# these ids, so this must stay deterministic: shards in sorted order, ids in encounter order.
# TAT-QA pages are rendered with the bundled DejaVu Sans Mono 2.35 (scripts/fonts/); another
# font gives different pixels for those pages.
def _prep(out):
    (out / "corpus_images").mkdir(parents=True, exist_ok=True)
    return out


def _write(d, corpus, queries, qrels):
    json.dump(corpus, open(d / "corpus_metadata.json", "w"))
    json.dump(queries, open(d / "queries.json", "w"))
    json.dump(qrels, open(d / "qrels.json", "w"))
    log(f"WROTE {d.name}: {len(corpus)} corpus, {len(queries)} queries, {len(qrels)} qrels")


# ---------------- ColPali train set (content-deduped pages) ----------------
def build_colpali(raw, out):
    """Every row's query is answered by its own page. Pages are deduped by decoded-pixel
    md5 (the same page recurs under different filenames). Native train/test split kept."""
    d = _prep(out)
    files = sorted(glob.glob(str(raw / "**" / "*.parquet"), recursive=True))
    corpus, queries, qrels, hash_to_cid = [], [], [], {}
    for f in files:
        split = "test" if os.path.basename(f).lower().startswith("test") else "train"
        for row in pq.read_table(f).to_pylist():
            fname, img, query = row.get("image_filename"), row.get("image"), row.get("query")
            src = row.get("source") or "unknown"
            if fname is None or not (isinstance(img, dict) and img.get("bytes") is not None):
                continue
            rgb = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            chash = hashlib.md5(f"{rgb.size}".encode() + rgb.tobytes()).hexdigest()
            if chash not in hash_to_cid:
                cid = len(corpus)
                hash_to_cid[chash] = cid
                rel = f"corpus_images/{cid}.png"
                rgb.save(d / rel)
                corpus.append({"corpus_id": cid, "image_path": rel, "doc_id": f"{src}/{fname}",
                               "source": src, "image_filename": fname, "aliases": []})
            else:
                cid = hash_to_cid[chash]
                alias = [src, fname]
                if alias != [corpus[cid]["source"], corpus[cid]["image_filename"]] \
                        and alias not in corpus[cid]["aliases"]:
                    corpus[cid]["aliases"].append(alias)
            if not query or not str(query).strip():
                continue
            qid = len(queries)
            queries.append({"query_id": qid, "query": str(query).strip(), "source": src,
                            "split": split, "answer": row.get("answer")})
            qrels.append({"query_id": qid, "corpus_id": cid, "score": 1})
        log(f"  processed {Path(f).name}: corpus={len(corpus)} queries={len(queries)}")
    _write(d, corpus, queries, qrels)


# ---------------- DocVQA ----------------
def build_docvqa(raw, out, splits):
    d = _prep(out)
    files = []
    for sp in splits:
        files += sorted(glob.glob(str(raw / f"DocVQA/{sp}-*.parquet")))
    corpus, queries, qrels, seen = [], [], [], set()
    for fp in files:
        for batch in pq.ParquetFile(fp).iter_batches(batch_size=256):
            b = batch.to_pydict()
            for i in range(len(b["questionId"])):
                img = b["image"][i]
                cid = Path(img["path"]).stem
                if cid not in seen:
                    seen.add(cid)
                    with open(d / "corpus_images" / f"{cid}.png", "wb") as f:
                        f.write(img["bytes"])
                    corpus.append({"corpus_id": cid, "image_path": f"corpus_images/{cid}.png"})
                qid = str(b["questionId"][i])
                queries.append({"query_id": qid, "query": b["question"][i]})
                qrels.append({"query_id": qid, "corpus_id": cid, "score": 1})
        log(f"  processed {Path(fp).name}: corpus={len(corpus)} queries={len(queries)}")
    _write(d, corpus, queries, qrels)


# ---------------- VisRAG in-domain (pages deduped by image-byte md5) ----------------
# Sub-sources whose license forbids redistribution: their rows still advance the id
# counters (so every other query/page keeps the id the manifest uses), but no image,
# query or qrel is written.
VISRAG_EXCLUDE = {"SlideVQA"}


def build_visrag(raw, out):
    d = _prep(out)
    files = sorted(glob.glob(str(raw / "data/train-*.parquet")))
    corpus, queries, qrels, img_hash_to_cid, qn = [], [], [], {}, 0
    skipped = 0
    for fp in files:
        for batch in pq.ParquetFile(fp).iter_batches(batch_size=256):
            b = batch.to_pydict()
            for i in range(len(b["query"])):
                src = b.get("source", [None] * len(b["query"]))[i]
                excluded = src in VISRAG_EXCLUDE
                raw_img = b["image"][i]["bytes"]
                h = hashlib.md5(raw_img).hexdigest()
                cid = img_hash_to_cid.get(h)
                if cid is None:
                    cid = f"img{len(img_hash_to_cid)}"
                    img_hash_to_cid[h] = cid
                    if not excluded:
                        with open(d / "corpus_images" / f"{cid}.png", "wb") as f:
                            f.write(raw_img)
                        corpus.append({"corpus_id": cid, "image_path": f"corpus_images/{cid}.png"})
                qid = f"q{qn}"; qn += 1
                if excluded:
                    skipped += 1
                    continue
                queries.append({"query_id": qid, "query": b["query"][i], "source": src})
                qrels.append({"query_id": qid, "corpus_id": cid, "score": 1})
        log(f"  processed {Path(fp).name}: corpus={len(corpus)} queries={len(queries)}")
    log(f"  visrag: skipped {skipped} rows from {sorted(VISRAG_EXCLUDE)} (ids preserved)")
    _write(d, corpus, queries, qrels)


# ---------------- TAT-QA (table + paragraphs rendered to a page image) ----------------
_FONT_CACHE = {}
_FONT_PATH = Path(__file__).resolve().parent / "fonts" / "DejaVuSansMono.ttf"


def _font(sz):
    if sz not in _FONT_CACHE:
        _FONT_CACHE[sz] = ImageFont.truetype(str(_FONT_PATH), sz)
    return _FONT_CACHE[sz]


def _render_tatqa(table_rows, paragraphs):
    F = _font(14); pad = 8; line_h = 20
    ncol = max((len(r) for r in table_rows), default=1)
    col_w = [0] * ncol
    for r in table_rows:
        for j, c in enumerate(r):
            col_w[j] = max(col_w[j], len(str(c)))
    col_px = [min(max(w, 3), 40) * 8 + 2 * pad for w in col_w]
    tbl_w = sum(col_px) if col_px else 400
    tbl_h = len(table_rows) * line_h + pad
    para_text = "\n".join(textwrap.fill(p.get("text", ""), width=max(tbl_w // 8, 60)) for p in paragraphs)
    para_lines = para_text.count("\n") + 1
    W = max(tbl_w, 800) + 2 * pad
    H = tbl_h + para_lines * line_h + 4 * pad + line_h
    img = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(img)
    y = pad
    for r in table_rows:
        x = pad
        for j in range(ncol):
            cell = str(r[j]) if j < len(r) else ""
            dr.rectangle([x, y, x + col_px[j], y + line_h], outline="black")
            dr.text((x + pad, y + 2), cell[:col_px[j] // 8], fill="black", font=F)
            x += col_px[j]
        y += line_h
    y += line_h
    dr.text((pad, y), para_text, fill="black", font=F)
    return img


def build_tatqa(raw, out, names):
    d = _prep(out)
    corpus, queries, qrels = [], [], []
    for name in names:
        data = json.load(open(raw / f"tatqa_dataset_{name}.json"))
        log(f"tatqa {name}: {len(data)} instances")
        for inst in data:
            tbl = inst["table"]
            cid = tbl["uid"]
            _render_tatqa(tbl["table"], inst.get("paragraphs", [])).save(d / "corpus_images" / f"{cid}.png")
            corpus.append({"corpus_id": cid, "image_path": f"corpus_images/{cid}.png"})
            for q in inst["questions"]:
                queries.append({"query_id": q["uid"], "query": q["question"]})
                qrels.append({"query_id": q["uid"], "corpus_id": cid, "score": 1})
    _write(d, corpus, queries, qrels)


# ---------------- REAL-MM-RAG / InfoVQA ({image, image_filename, query} parquet) ----------------
def build_imgquery(raw, out, splits):
    """Corpus = unique pages keyed by image_filename (includes corpus-only distractor
    pages); one query per non-null query / paraphrase column."""
    d = _prep(out)
    files = []
    for sp in splits:
        files += sorted(glob.glob(str(raw / f"data/{sp}-*.parquet")))
    corpus, queries, qrels, seen, qn = [], [], [], {}, 0
    for fp in files:
        for batch in pq.ParquetFile(fp).iter_batches(batch_size=128):
            b = batch.to_pydict()
            for i in range(len(b["image_filename"])):
                fn = b["image_filename"][i]
                if fn not in seen:
                    cid = str(len(seen))
                    seen[fn] = cid
                    Image.open(io.BytesIO(b["image"][i]["bytes"])).convert("RGB") \
                        .save(d / "corpus_images" / f"{cid}.png")
                    corpus.append({"corpus_id": cid, "image_path": f"corpus_images/{cid}.png"})
                cid = seen[fn]
                for col in QUERY_COLS:
                    if col not in b:
                        continue
                    text = b[col][i]
                    if text is None or not str(text).strip():
                        continue
                    qid = f"q{qn}"; qn += 1
                    queries.append({"query_id": qid, "query": str(text).strip()})
                    qrels.append({"query_id": qid, "corpus_id": cid, "score": 1})
    _write(d, corpus, queries, qrels)


# subset -> (raw source dir, builder)
RECIPES = {
    "colpali":          ("colpali_train_set", lambda r, o: build_colpali(r, o)),
    "docvqa_train":     ("docvqa",            lambda r, o: build_docvqa(r, o, ["train", "validation"])),
    "visrag_in_domain": ("visrag_in_domain",  lambda r, o: build_visrag(r, o)),
    "tatqa_train":      ("tatqa",             lambda r, o: build_tatqa(r, o, ["train", "dev"])),
    "infovqa_train":    ("infovqa_train",     lambda r, o: build_imgquery(r, o, ["train"])),
    "realmm_finreport": ("realmm_finreport",  lambda r, o: build_imgquery(r, o, ["train", "test"])),
    "realmm_finslides": ("realmm_finslides",  lambda r, o: build_imgquery(r, o, ["train", "test"])),
}


def prepare(root, only):
    for subset, (src, fn) in RECIPES.items():
        if not only or subset in only:
            log(f"=== {subset} <- raw/{src}")
            fn(root / "raw" / src, root / "prepared" / subset)


# =============================== training set: assemble ===============================
def assemble(root, manifests, out, all_negatives=False, upweight=False):
    """Manifest(s) + prepared corpora -> the training file train.py reads:
    {"config": {...}, "queries": {"<i>": {query, split, source, orig_qid, source_lang, sub_source,
     bucket, candidates: [{corpus_id, image_path, label, qrels_score, rank, retrieval_score,
     in_top_k, is_fallback}]}}}
    Query text comes from the manifest (translations, openly-licensed English) or from
    prepared/<source>/queries.json (checked against query_sha1). A group = the positive(s) + the
    negatives flagged used_in_training (all 50 with all_negatives). One group per manifest line,
    as in the paper, in the paper's order (train_index); upweight=True repeats each line
    train_copies times (finance sources 2x).
    image_path is relative to root; rank is 0-indexed (the manifest's is 1-indexed)."""
    prep = root / "prepared"
    texts, paths = {}, {}

    def load_source(src):
        if src in texts:
            return
        d = prep / src
        texts[src] = {str(q["query_id"]): q["query"] for q in json.load(open(d / "queries.json"))}
        paths[src] = {f"{src}:{c['corpus_id']}": str((d / c["image_path"]).relative_to(root))
                      for c in json.load(open(d / "corpus_metadata.json"))}

    lines = []
    for manifest in manifests:
        opener = gzip.open if str(manifest).endswith(".gz") else open
        with opener(manifest, "rt") as f:
            lines += [json.loads(line) for line in f]
    if all("train_index" in r for r in lines):   # the order the paper model was trained in
        lines.sort(key=lambda r: r["train_index"])
    Q, stats = {}, Counter()
    for r in lines:
        src = r["source"]
        load_source(src)
        text = r["query"]
        if text is None:
            text = texts[src][r["orig_qid"]]
            if hashlib.sha1(text.encode()).hexdigest() != r["query_sha1"]:
                stats["query_text_mismatch"] += 1
        cands = [{"corpus_id": p["corpus_id"], "image_path": paths[src][p["corpus_id"]],
                  "label": 1, "qrels_score": 1,
                  "rank": None if p["rank"] is None else p["rank"] - 1,
                  "retrieval_score": p["score"], "in_top_k": p["rank"] is not None,
                  "is_fallback": p["fallback"]} for p in r["positives"]]
        cands += [{"corpus_id": n["corpus_id"], "image_path": paths[src][n["corpus_id"]],
                   "label": 0, "qrels_score": 0, "rank": n["rank"] - 1,
                   "retrieval_score": n["score"], "in_top_k": True, "is_fallback": False}
                  for n in r["negatives"] if all_negatives or n["used_in_training"]]
        rec = {"query": text, "split": r["split"], "source": src, "orig_qid": r["orig_qid"],
               "source_lang": None if r["lang"] == "english" else r["lang"],
               "sub_source": r["sub_source"], "bucket": r["bucket"], "candidates": cands}
        for _ in range(r["train_copies"] if upweight else 1):
            Q[str(len(Q))] = rec
        stats[f"lines_{r['split']}"] += 1

    cfg = {"built_from": [Path(m).name for m in manifests], "all_negatives": all_negatives, "upweight": upweight,
           "groups": len(Q), **stats}
    json.dump({"config": cfg, "queries": Q}, open(out, "w"), ensure_ascii=False)
    log(json.dumps(cfg))
    if stats["query_text_mismatch"]:
        log(f"WARNING: {stats['query_text_mismatch']} English queries differ from the release "
            f"(different upstream revision?)")
    log(f"wrote {out}")


# =============================== benchmarks ===============================
def vidore_v3(root, subsets):
    """vidore/vidore_v3_{subset} -> {root}/vidore_v3/{subset}/ (all 6 query languages kept)."""
    for s in subsets:
        raw = root / "raw" / f"vidore_v3_{s}"
        hub_download(f"vidore/vidore_v3_{s}", VIDORE_V3[s], ["corpus/*", "queries/*", "qrels/*"], raw)
        rows = lambda cfg: [r for f in sorted(glob.glob(str(raw / cfg / "*.parquet")))  # noqa: E731
                            for r in pq.read_table(f).to_pylist()]
        out = root / "vidore_v3" / s
        (out / "corpus_images").mkdir(parents=True, exist_ok=True)
        meta = []
        for r in rows("corpus"):
            cid, dst = r["corpus_id"], out / "corpus_images" / f"{r['corpus_id']}.png"
            if not dst.exists():
                Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(dst, "PNG")
            meta.append({"corpus_id": cid, "doc_id": r.get("doc_id"),
                         "page_number_in_doc": r.get("page_number_in_doc"),
                         "image_path": f"corpus_images/{cid}.png", "markdown": r.get("markdown")})
        queries, qrels = rows("queries"), rows("qrels")
        _write(out, meta, queries, qrels)


def unidoc(root, domains):
    """Salesforce/UniDoc-Bench -> {root}/unidoc/{domain}/. Corpus = every page shipped for the
    domain; qrels = gt_image_paths. Pages that fail to decode upstream are dropped (one in
    energy, 3626869_page_0014, which is in no query's qrels)."""
    raw = root / "raw" / "unidoc"
    for dom in domains:
        hub_download(*UNIDOC, [f"images/{dom}/*", f"data/{dom}-*.parquet"], raw, workers=16)
        out = root / "unidoc" / dom
        (out / "corpus_images").mkdir(parents=True, exist_ok=True)
        meta = []
        for p in sorted((raw / "images" / dom).glob("*/*.png")):
            try:
                Image.open(p).load()
            except Exception:  # noqa: BLE001
                log(f"[{dom}] dropping undecodable page {p.name}")
                continue
            dst = out / "corpus_images" / p.name
            if not dst.exists():
                os.link(p, dst)
            doc_id, _, page = p.stem.rpartition("_page_")
            meta.append({"corpus_id": p.stem, "doc_id": doc_id, "page_number_in_doc": int(page),
                         "image_path": f"corpus_images/{p.name}"})
        ids = {m["corpus_id"] for m in meta}
        queries, qrels = [], []
        for qid, r in enumerate(pq.read_table(next((raw / "data").glob(f"{dom}-*.parquet"))).to_pylist()):
            queries.append({"query_id": qid, "query": r["question"], "answer": r["answer"],
                            "question_type": r["question_type"], "answer_type": r["answer_type"],
                            "language": "english"})
            qrels += [{"query_id": qid, "corpus_id": Path(g).stem, "score": 1}
                      for g in dict.fromkeys(r["gt_image_paths"]) if Path(g).stem in ids]
        _write(out, meta, queries, qrels)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["download", "prepare", "assemble", "vidore_v3", "unidoc"])
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--only", nargs="*", help="download/prepare: sources; vidore_v3/unidoc: subsets")
    ap.add_argument("--manifest", nargs="+",
                    help="assemble: train_manifest_en.jsonl.gz and/or train_manifest_ml.jsonl.gz")
    ap.add_argument("--out", help="assemble: output file (default <data_dir>/train_set.json)")
    ap.add_argument("--all_negatives", action="store_true", help="assemble: keep all 50 negatives")
    ap.add_argument("--upweight", action="store_true",
                    help="assemble: repeat lines train_copies times (not used in the paper)")
    a = ap.parse_args()
    root = Path(a.data_dir)
    if a.step == "download":
        download(root, a.only)
    elif a.step == "prepare":
        prepare(root, a.only)
    elif a.step == "assemble":
        if not a.manifest:
            ap.error("assemble needs --manifest")
        assemble(root, a.manifest, Path(a.out) if a.out else root / "train_set.json",
                 a.all_negatives, a.upweight)
    elif a.step == "vidore_v3":
        vidore_v3(root, a.only or list(VIDORE_V3))
    else:
        unidoc(root, a.only or UNIDOC_DOMAINS)
    log("done")


if __name__ == "__main__":
    main()
