# MQ-MINER

**Multilingual Queries with Mined Hard Negatives for Visual Document Reranking**: a multilingual
training set with mined hard negatives, and the reranker trained on it
(Eagle2-1B + LoRA + linear head). `weights/` holds the checkpoint, `manifest/` the training set
(EN + fr/es/it/de/pt), `runs/` the first-stage retrieval runs used in the paper.

## Setup
```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt   # CUDA 12.8
```
The base model `nvidia/Eagle2-1B` is downloaded and patched on first use.

## Evaluate
```bash
python scripts/data.py vidore_v3 --data_dir data      # or: unidoc
python scripts/evaluate.py --data_dir data/vidore_v3 --runs runs/webai/vidore_v3 --depth 30 --out results/webai_vidore_v3
```
Use `runs/colqwen2/...` for the second retriever and `torchrun --nproc_per_node=N` for several GPUs.

## Manifest
`manifest/train_manifest_en.jsonl.gz` (116,656 English queries) and `train_manifest_ml.jsonl.gz`
(121,137 fr/es/it/de/pt translations), one JSON line per query:

| field | meaning |
|---|---|
| `id` | `<source>:<orig_qid>:<lang>` |
| `source`, `sub_source`, `orig_qid` | upstream dataset and its query id |
| `lang` | `english`, `french`, `spanish`, `italian`, `german` or `portuguese` |
| `query`, `query_sha1` | query text; `null` for English queries whose license forbids redistribution (rebuilt by `data.py`, checked against `query_sha1`) |
| `positives` | the gold page: `corpus_id`, its webAI `rank` and `score`; `rank` is `null` and `fallback` true if it was not in the top 100 |
| `negatives` | 50 mined hard negatives (`corpus_id`, `rank`, `score`); `used_in_training` marks the ones used for the released checkpoint (15, or fewer for 5% of VisRAG queries after SlideVQA was removed) |
| `bucket` | difficulty by gold rank: `b0` 1, `b1` 2–3, `b2` 4–10, `b3` 11–50, `b4` 51–100, `fallback` >100 |
| `train_index` | position in the training order of the released checkpoint |
| `train_copies` | 2 for some TAT-QA / REAL-MM-RAG (finance) queries, repeated by `assemble --upweight` (not used in the paper) |
| `split` | always `train` |

`corpus_id` is `<source>:<page id>`, the page id in `data/prepared/<source>/corpus_metadata.json`;
ranks are 1-based webAI-ColVec1-4b ranks.

## Rebuild the training set and train
```bash
python scripts/data.py download --data_dir data       # 7 sources, ~77 GB
python scripts/data.py prepare  --data_dir data       # ~85 GB of page images
python scripts/data.py assemble --data_dir data \
    --manifest manifest/train_manifest_en.jsonl.gz manifest/train_manifest_ml.jsonl.gz
torchrun --nproc_per_node=2 scripts/train.py --train_set data/train_set.json --data_dir data \
    --output_dir checkpoints/mq-miner                 # paper: 1 epoch, released weights = step 9500
```

## License
Code MIT; checkpoint CC BY-NC 4.0 (from Eagle2-1B); annotations CC BY 4.0. Query text, including
translations, follows the license of its source; see [LICENSE.md](LICENSE.md).
