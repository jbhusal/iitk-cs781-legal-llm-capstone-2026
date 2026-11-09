# IITK CS781 Legal LLM Capstone (2026)

Code for my CS781 (IIT Kanpur) capstone: four legal language models (A, B, C, D) pretrained from scratch in Phase 1, then adapted to six legal NLP tasks (NER, rhetorical roles, judgment prediction, statute identification, summarisation, case retrieval) using LoRMA, QLoRA, IA³, adapters and knowledge distillation. Kaggle 2× T4, official data only.

**Author:** Jatin Bhusal · **Course:** CS781, IIT Kanpur · **Competition:** CS-781 Capstone on Codabench · **Data:** [Exploration-Lab/CS-781-Capstone](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone)

---

## Results at a glance

Official Codabench scores on the hidden test sets. Each phase README lists every submission file with its score.

| Phase | Task | Metric | Best submission | Official score |
|---|---|---|---|---|
| [1](Phase%201/) | Legal language modelling (128-word continuation) | BERTScore F1 | Model D (encoder-decoder) + MBR decoding | **0.8389** |
| [2](Phase%202/) | Legal named entity recognition | Strict macro-F1 | `sub-1.zip` (best model on validation: C + LoRMA) | **0.5664** |
| [3](Phase%203/) | Rhetorical role prediction | Macro-F1 | E1: D encoder + LoRMA + BiLSTM over sentences | **0.5242** |
| [4](Phase%204/) | Court judgment prediction | Macro-F1 | S6: mean-logit ensemble of five runs (all four models) | **0.7798** |
| [5](Phase%205/) | Legal statute identification | Macro-F1 | S6: half-depth D student distilled from two teachers | **0.3681** |
| [6](Phase%206/) | Abstractive summarisation | BERTScore F1 | S1: D encoder-decoder + LoRMA, plan-then-write | **0.8528** |
| [7](Phase%207/) | Prior case retrieval | Micro-F1@7 | S7: ensemble of six dual-encoder retrievers | **0.3479** |

---

## Repository layout

```
.
├── README.md
├── LICENSE
├── Phase 1/                         pretraining the four models
│   ├── p1_00_data_prep.ipynb        corpus cleaning + local dev set
│   ├── p1_01_tokenizer.ipynb        T1 (BPE) and T2 (SentencePiece) tokenizers
│   ├── p1_02_bench.ipynb            GPU kernel / throughput benchmark
│   ├── modelA/                      trainer, predictor, submission.zip
│   ├── modelB/                      trainer, predictor, submission.zip
│   ├── modelC/                      trainer, predictor, submission.zip
│   └── modelD/                      trainer, predictor, continued training, submission.zip   ← best
├── Phase 2/                         NER: 3 notebooks + sub-1 … sub-4.zip
├── Phase 3/                         rhetorical roles: 2 notebooks + submission1 … 6.zip
├── Phase 4/                         judgment prediction: 2 notebooks + build_s6.py + 9 submission zips
├── Phase 5/                         statute identification: 1 notebook + 9 submission zips
├── Phase 6/                         summarisation: 1 notebook + submission1 … 7.zip
└── Phase 7/                         case retrieval: 1 notebook + 8 submission zips
```

Every phase folder has a `README.md` with:
- the task and metric;
- data handling;
- the model and method;
- every run and its settings;
- every submission file with its official score;
- the best model and how to run it.

All code is in the notebooks. Each one writes its own Python modules (e.g. `legal_d.py`, `pcr_lib.py`) with `%%writefile` cells, so one notebook is enough to run a phase. Phase 4 also has `build_s6.py` for its ensemble.

**Submission files.** Every `*.zip` holds one `prediction.json`: exactly what was uploaded to Codabench, or a run's output in the same format.

**Not included.** Model weights and checkpoints, and the competition data. The notebooks load checkpoints from Kaggle datasets and download the data from Hugging Face at run time.

---

## The four Phase-1 models

All four were trained from random initialisation on the official Phase-1 legal corpus only. They cover four architecture families, so they differ in how attention flows, not just in size. Every later phase starts from these checkpoints.

| | A | B | C | D |
|---|---|---|---|---|
| Family | Decoder-only (Llama-style) | Efficient decoder (local + global) | Prefix-LM | Encoder-decoder (T5-style) |
| Parameters | ~163M | 121.8M | 112.8M | 194.5M (encoder ~100M) |
| Shape | 16 layers × 896 | 24 layers × 640 | 14 layers × 768 | 12 encoder + 12 decoder layers × 768 |
| Attention | GQA 14q / 2kv, RoPE | Sliding window 512 + a global layer every 6th, MQA, QK-norm | GQA 12q / 4kv, QK-norm, bidirectional over the prompt | GQA 12q / 4kv, QK-norm; RoPE in self-attention, none in cross-attention |
| Feed-forward | SwiGLU, RMSNorm, no biases | SwiGLU, RMSNorm | GELU with biases, RMSNorm | SwiGLU, RMSNorm, no biases |
| Tokenizer | T1: byte-level BPE, 32k | T1 | T2: SentencePiece Unigram, 32k | T2 |
| Training text | 52M tokens | ~4 h run | 448M tokens | 650M tokens |
| Phase-1 official | 0.8231 | 0.8326 | 0.8362 | 0.8389 (MBR) |

Checkpoints used by Phases 2–7: `A_step798_tokens52297728.pt`, the final `B_step*` checkpoint, `C_step13683_tokens448364544.pt`, `D_step10585_tokens650342400.pt`.

---

## Techniques implemented

All adaptation methods are written from scratch in PyTorch. No PEFT library is used.

| Technique | Phases | In one line |
|---|---|---|
| **LoRMA** (multiplicative low-rank) | 2, 3, 4, 5, 6, 7 | W' = (I + (α/r)·B·A)·W₀ with B = 0 at start, so training begins exactly at the Phase-1 model |
| LoRA / rsLoRA | 2, 3, 6 | additive low-rank update; rsLoRA scales by α/√r for high ranks |
| QLoRA | 5, 6, 7 | frozen base stored in our own NF4 4-bit format with double-quantised scales + LoRA |
| AdaLoRA | 4 | SVD-style low-rank update whose rank budget moves to the most important layers |
| IA³ | 3, 7 | learned scaling vectors on keys, values and the MLP input |
| BitFit | 4 | only bias terms train |
| Houlsby adapters | 5, 6 | small bottleneck layers after attention and MLP |
| Prompt / prefix tuning | 3, 5, 6, 7 | learned vectors at the input (prompt) or at every layer (prefix) |
| Partial / layer-wise fine-tuning, gradual unfreezing | 4, 5, 7 | train only the top blocks, with smaller learning rates lower down |
| Full fine-tuning | 2, 3, 4, 5, 6, 7 | reference runs and distillation students |
| **Knowledge distillation** | 3, 4, 5, 6, 7 | a smaller student learns the teacher's soft outputs (and, in Phase 4, the teacher's attention) |
| Ensembles / consensus | 4, 5, 6, 7 | mean of logits or z-scored scores; in Phase 6, segment-level MBR consensus |
| MBR decoding | 1 | pick the sampled continuation that agrees most with the others |

---

## Setup (all phases)

- **Platform:** Kaggle notebooks, **GPU T4 × 2** (Phase 2: one T4 is enough), internet on.
- **Secrets:** `HF`, a Hugging Face token that can read the competition dataset. The Phase-1 trainers and tokenizer notebook also use `KAGGLE_API_TOKEN` to publish checkpoints and tokenizers as Kaggle datasets.
- **Inputs to attach:**
  - the tokenizer dataset (`t1_bpe.json`, `t2_unigram.model`), made by `Phase 1/p1_01_tokenizer.ipynb`;
  - the Phase-1 checkpoint datasets. File names must start with `A_step`, `B_step`, `C_step` or `D_step`; `.pt`, `.zip` or a folder that Kaggle unpacked all work;
  - for Phase 2, the Phase-2 data files.
- **Input paths:** Phases 3–7 find their inputs by searching `/kaggle/input`. A few Phase-1 notebooks point to the author's Kaggle dataset paths (`/kaggle/input/datasets/<user>/…`); change those lines to your own dataset paths.
- **Software:** Kaggle's default Python image (PyTorch, NumPy, SciPy, scikit-learn, `huggingface_hub`) plus `sentencepiece` and `tokenizers`. Phases 1 and 6 also use `bert-score` (roberta-large) for local scoring only. It is never used for training or for generating outputs.
- **Running:** open the notebook, attach the inputs, then **Save Version → Save & Run All**. Each notebook runs its self-tests first, then prints its training log, a results table and the path of every `submission.zip`.

---

## Reproducibility

Re-run the notebooks to obtain results; scores will differ slightly from those reported here. Seeds are fixed, but GPU computation is not deterministic (fp16 mixed precision, cuDNN kernels, atomic operations). Many runs also stop on a wall-clock time budget, so the number of training steps depends on GPU speed. Kaggle's software image also changes over time.

---

## Data, licence and use of tools

- Only the official competition data was used: no external legal corpora, statutes or scraped judgments.
- The data is not redistributed here; see its terms on Hugging Face.
- Code: MIT licence (see `LICENSE`).
- LLM usage is declared in the final project report, as the course requires.
