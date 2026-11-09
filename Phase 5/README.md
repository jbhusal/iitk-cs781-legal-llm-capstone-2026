# Phase 5 — Legal statute identification (LSI)

**Task.** Map each legal document to the **set** of statutes that apply, out of 100 statute ids (multi-label).
**Metric.** Macro-F1 over statutes, so a rare statute counts as much as a frequent one.
**Best submission.** S6, a half-depth D student distilled from S1 + S3: **0.3681 macro-F1** on Codabench.

---

## Files

| File | What it is |
|---|---|
| `p5_lsi_all_runs.ipynb` | One notebook with every run (S1–S7, R1, R2, TF-IDF). Writes `legal_d.py`, `c_model.py`, `a_model.py`, `b_model.py`, `lsi_lib.py` (pipeline, methods, self-tests), `lsi_train.py` (one run per process) and `lsi_tfidf.py` (CPU baseline) |
| `S1.zip`, `s2.zip`, `S3.zip` … `S9.zip` | The nine Codabench submissions. The upload names follow the submit order, not the run names; the mapping is in the Results table |

The best model (S6) and its two teachers (S1, S3) are all inside this notebook.

---

## Data

| Split | Raw | Cleaning | Documents used | Statutes per document |
|---|---|---|---|---|
| Train | 42,750 | 324 texts also in dev/test dropped; 277 duplicate texts merged (60 with different statute sets: union kept) | 42,149 | mean 3.70; 30.8% have 5+ |
| Dev | 10,181 | none needed | 10,181 | mean 3.92 |
| Test | 13,019 | ids unique | 13,019 | — |

- **Statute counts:** every statute has at least 268 training documents; the most frequent has 11,514.
- **Masking:** statute mentions are masked in the text (`<SECTION>`, `<ACT>`, `<ENTITY>`).
- **Length:** documents have a median of 434 words (645 T2 tokens, 3.52 chunks on average).
- **Train split:** 33,714 documents adapt the backbone, 7,173 train the readers, 1,262 choose the reader; stratified by each document's rarest statute.
- **Ids:** internal ids are `<split>:<id>`, so ids repeated across splits never collide; the raw id is used in the submission.

---

## Method

Documents are cut into 512-token chunks with a 64-token overlap; at most 12 per document, spread evenly.

1. **Adapt.** The Phase-1 backbone with its technique, topped by a **label-wise attention head**: each statute has its own attention over the chunk's tokens (CAML / LAAT style). A document is a bag of up to 3 random chunks, and its statute scores are the **log-sum-exp pool** of the chunk scores. Weighted BCE; best checkpoint by validation macro-average-precision.
2. **Encode.** Every chunk of the held-out train documents, dev and test becomes a vector plus statute scores.
3. **Read.** Six readers run over the chunk vectors: mean, max, attention, label-wise attention, label-wise attention + statute co-occurrence, and transformer + label-wise attention. A blend of the best reader with the stage-1 scores is also tried. The choice is made on validation macro-AP. A **loss study** (BCE / weighted BCE / focal / asymmetric) and a **section study** (first chunk; first / middle / last third) reuse the cached vectors.
4. **Thresholds.** Four strategies: probability 0.5, one global threshold, one per statute, or per statute shrunk toward the global one. The strategy is chosen by an honest **2-fold check on dev** (fit on one half, score the other, both ways). Every document gets at least its top statute.

**Student (S6, R2):** encoder layers 0, 2, 4, 6, 8, 10 of D (62.3M), trained end to end with a label-wise attention reader over ≤ 4 chunks. S6 adds the teachers' (S1 + S3) mean logit as a soft target (α 0.5, T 2).

**Methods** (own code):
- **LoRMA:** (I + s·B·A)·W₀.
- **QLoRA:** our own NF4: each weight is one of 16 normal-distribution quantiles in blocks of 64, and the block scales are stored as int8 (double quantisation); plus LoRA r 16.
- **Houlsby adapters:** bottleneck 64 after attention and MLP, plus norms.
- **Deep prefix tuning:** 16 learned states at every block, started from real activations.
- **Partial fine-tuning:** top 6 of 12 layers, layer-wise learning-rate decay 0.85.

---

## Runs and settings

Common: 2 epochs (capped by each run's budget), head learning rate 1e-3, gradient checkpointing, seed 13; 1,000 validation documents.

| Run | Backbone | Technique | Settings |
|---|---|---|---|
| S1 | D encoder (100.1M) | **LoRMA**, every attention + MLP layer | r 16, α 32, lr 5e-4, 55 min; KD teacher |
| S2 | B (121.8M) | **QLoRA** (NF4 4-bit + LoRA) | r 16, α 32, lr 5e-4, 60 min |
| S3 | C (112.8M) | **Houlsby adapters** | bottleneck 64, lr 1e-3, 55 min; KD teacher |
| S4 | A (162.7M) | **Prefix tuning** (deep) | 16 prefix states, lr 3e-3, 60 min |
| S5 | D encoder | **Partial / layer-wise fine-tuning** | top 6 layers, lr 1e-4, decay 0.85, 50 min |
| **S6** | half of D (62.3M) | **Knowledge distillation** from S1 + S3 | lr 1e-4, 60 min, α 0.5, T 2 |
| S7 | S1–S6 | Ensemble: mean probability, thresholds re-fitted | CPU |
| R1 | D encoder | Full fine-tuning (reference for S1) | lr 3e-5, 45 min |
| R2 | same student as S6 | Gold labels only (control for S6) | lr 1e-4, 60 min |
| TF-IDF | none | Word 1–2-gram TF-IDF + one logistic regression per statute | CPU, runs alongside |

---

## Results

"2-fold" means thresholds fitted on one half of dev and scored on the other (the fair comparison). "Fit" means thresholds fitted on all of dev, which is what the submission uses.

| Run | File | Trainable % | Storage (MB) | Dev macro-F1 (2-fold) | Dev macro-F1 (fit) | Dev micro-F1 | Official macro-F1 |
|---|---|---|---|---|---|---|---|
| **S6 KD student** | `S1.zip` | 100 | 119 | 0.3695 | 0.3842 | 0.4772 | **0.3681** |
| R2 student, gold only | `s2.zip` | 100 | 119 | 0.3637 | 0.3767 | 0.4655 | 0.3603 |
| S7 ensemble | `S9.zip` | — | — | 0.3594 | 0.3711 | 0.4621 | 0.3555 |
| S3 C + adapters | `S3.zip` | 2.48 | 220 | 0.3584 | 0.3700 | 0.4629 | 0.3526 |
| S1 D + LoRMA | `S4.zip` | 2.65 | 196 | 0.3318 | 0.3443 | 0.4366 | 0.3293 |
| R1 D full fine-tuning | `S5.zip` | 100 | 191 | 0.3273 | 0.3401 | 0.4365 | 0.3262 |
| S2 B + QLoRA | `S6.zip` | 3.49 | 97 | 0.3253 | 0.3386 | 0.4314 | 0.3262 |
| S5 D partial FT | `S7.zip` | 37.72 | 191 | 0.3252 | 0.3395 | 0.4316 | 0.3219 |
| S4 A + prefix | `S8.zip` | 0.14 | 310 | 0.2431 | 0.2548 | 0.3467 | 0.2407 |
| TF-IDF + LR | — | — | — | 0.3986 | 0.4196 | 0.5054 | not submitted |

The TF-IDF row is a reference only: every submission must start from a Phase-1 model.

**Best model:** S6, the 62.3M distilled student, at 29% of its two teachers' parameters (212.9M).

---

## How to run

1. Inputs: the tokenizer dataset (`t1_bpe.json`, `t2_unigram.model`) and the A, B, C and D checkpoint datasets.
2. Secret `HF`; **GPU T4 × 2**; internet on.
3. **Save Version → Save & Run All.** The real run took 6.6 h. The scheduler runs two jobs at a time, makes S6 wait for S1 and S3, and skips any run that could not finish before 11 h.
4. Outputs: `p5_out/S1 … S7/submission.zip` and `p5_out/report_files.zip` (all tables).

The notebook prints, per run:
- the reader comparison, loss study, section study and threshold strategies;
- per-statute F1 by frequency band;
- the distillation comparison;
- the QLoRA weight error.

---

## Reproducibility

Re-run the notebooks to obtain results; scores will differ slightly from those reported here. Seeds are fixed, but GPU computation is not deterministic (fp16 mixed precision, cuDNN kernels, atomic operations). Every run also stops on a wall-clock time budget, so the number of training steps depends on GPU speed. Kaggle's software image also changes over time.
