# License

| Part | License |
|---|---|
| Code (`scripts/`) | MIT (below) |
| Reranker checkpoint (`weights/mq-miner.safetensors`) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), inherited from its base model [`nvidia/Eagle2-1B`](https://huggingface.co/nvidia/Eagle2-1B) (CC BY-NC 4.0) |
| MQ-MINER annotations (`manifest/`: mined negatives, ranks, scores, buckets, training flags) and `runs/` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the source content they refer to stays under its original license (table below) |
| Query text in `manifest/` (English where shipped, and all translations) | license of its source (table below) |
| Bundled font (`scripts/fonts/DejaVuSansMono.ttf`) | DejaVu Fonts License (`scripts/fonts/LICENSE-DejaVu`) |

## Upstream sources

Page images are never redistributed. English query text is included in the manifest only where
the source license allows it; otherwise `scripts/data.py` rebuilds it from the original dataset
at the pinned revision. EN = English queries in `train_manifest_en.jsonl.gz`.

| Source (Hub repo @ revision) | Sub-source | License | EN | English text in manifest |
|---|---|---|---:|:---:|
| [`vidore/colpali_train_set`](https://huggingface.co/datasets/vidore/colpali_train_set) @ `e13d359` | synthetic PDF | released without usage restrictions | 8,970 | yes |
| | TAT-DQA | CC BY 4.0 | 5,749 | yes |
| | DocVQA | RRC terms of use (docvqa.org) | 15,550 | no |
| | InfographicVQA | RRC terms of use (docvqa.org) | 2,341 | no |
| | ArxivQA | CC BY-SA 4.0 | 1,949 | no |
| [`openbmb/VisRAG-Ret-Train-In-domain-data`](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Train-In-domain-data) @ `9c2ecd9` | PlotQA | MIT | 27,442 | yes |
| | ArxivQA | CC BY-SA 4.0 | 7,445 | no |
| | InfoVQA | RRC terms of use | 4,607 | no |
| | MP-DocVQA | MIT (DocVQA pages, RRC terms) | 3,246 | no |
| | ChartQA | GPL-3.0 | 1,793 | no |
| | SlideVQA | no redistribution, **excluded** | 0 | – |
| [`lmms-lab/DocVQA`](https://huggingface.co/datasets/lmms-lab/DocVQA) @ `539088e` | DocVQA | mirror tagged Apache-2.0; RRC terms apply | 21,187 | no |
| [`vidore/infovqa_train`](https://huggingface.co/datasets/vidore/infovqa_train) @ `fd098df` | InfographicVQA | RRC terms of use | 2,659 | no |
| [`next-tat/TAT-QA`](https://huggingface.co/datasets/next-tat/TAT-QA) @ `c96247f` | TAT-QA | CC BY 4.0 | 7,166 | yes |
| [`ibm-research/REAL-MM-RAG_FinReport`](https://huggingface.co/datasets/ibm-research/REAL-MM-RAG_FinReport) @ `904a87d` | FinReport | CDLA-Permissive-2.0 | 2,943 | yes |
| [`ibm-research/REAL-MM-RAG_FinSlides`](https://huggingface.co/datasets/ibm-research/REAL-MM-RAG_FinSlides) @ `a4fafbf` | FinSlides | CDLA-Permissive-2.0 | 3,609 | yes |
| **Total** | | | **116,656** | |

Translations (fr/es/it/de/pt, `train_manifest_ml.jsonl.gz`, 121,137 queries) are included for all
sub-sources. Translated queries are derived from the source questions and are released under the
license of their source (table above); for sources with restrictive terms they are provided for
research use under those terms. All other MQ-MINER annotations (mined negatives, ranks, scores,
buckets, training flags) are released under CC BY 4.0.

| Benchmark / model | License | Use here |
|---|---|---|
| ViDoRe v3, 8 subsets (`vidore/vidore_v3_*`) | CC BY 4.0 | evaluation; downloaded by `scripts/data.py` |
| [`Salesforce/UniDoc-Bench`](https://huggingface.co/datasets/Salesforce/UniDoc-Bench) @ `2e22c59` | CC BY-NC 4.0 | evaluation; downloaded by `scripts/data.py` |
| [`webAI-Official/webAI-ColVec1-4b`](https://huggingface.co/webAI-Official/webAI-ColVec1-4b) | webAI Non-Commercial License v1.0 | produced `runs/webai/` and the mined negatives |
| [`vidore/colqwen2-v1.0`](https://huggingface.co/vidore/colqwen2-v1.0) | Apache-2.0 | produced `runs/colqwen2/` |

## MIT License (code)

Copyright (c) 2026 the MQ-MINER authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
