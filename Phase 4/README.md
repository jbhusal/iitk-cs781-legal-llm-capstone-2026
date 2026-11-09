# Phase 4 — Court judgment prediction (CJPE)

**Task.** Predict the outcome of a case (binary label 0/1) from the full judgment text.
**Metric.** Macro-F1.
**Best submission.** S6, a mean-logit ensemble of five runs covering all four Phase-1 models: **0.7798 macro-F1** on Codabench. Best single model: S1, D encoder + LoRMA (0.7693).

---

## Files

| File | Runs | Notes |
|---|---|---|
| `p4_01_cjpe_notebook1.ipynb` | S1, S2, S5, R1, R2 | D, C and the distilled student. Writes `legal_d.py`, `c_model.py`, `cj_lib.py`, `cj_train.py`, and `p4_nb1/ens_nb1.json` (dev/test logits of every run) |
| `p4_02_cjpe_notebook2.ipynb` | S3, S4, R3, TF-IDF | B and A. Writes `a_model.py`, `b_model.py`, `cj_lib.py`, `cj_train.py`, `cj_tfidf.py`, and `p4_nb2/ens_nb2.json` |
| `build_s6.py` | S6 | Ensemble of the five member runs from the two JSON files (CPU, seconds) |
| `S6_ensemble_submission.zip` | S6 | Best submission |
| `submissionS1.zip`, `submissionS2.zip`, `submissionS5.zip`, `submissionR1.zip`, `submissionR2.zip` | S1, S2, S5, R1, R2 | Submitted to Codabench |
| `submissionS3.zip`, `submissionS4.zip`, `submissionR3.zip` | S3, S4, R3 | Test predictions of runs not scored on Codabench (daily submission cap) |

The two notebooks are independent of each other: each downloads the data itself. All three files are needed for the best submission.

---

## Data

| Split | Raw rows | Cleaning | Cases used | Label 1 |
|---|---|---|---|---|
| Train | 37,387 | 5,734 duplicate copies dropped; none overlapped dev/test | 31,653 | 41.4% |
| Dev | 3,505 | 994 exact repeat rows (same id, text, label) dropped | 2,511 | 50.1% |
| Test | 1,517 | ids checked unique | 1,517 | — |

- **Length:** cases are long, median 2,030 words (90th percentile 6,431). With T2 that is a median of 2,452 tokens and 7.0 chunks per case.
- **Train split:** 25,322 cases to adapt the backbone, 5,382 to train the case readers, 949 to choose the reader; stratified by label.
- **Class weight:** train has fewer label-1 cases than dev, so the loss uses a positive-class weight of 1.41.

---

## Method (shared by every run except the students)

Each case is cut into 512-token chunks with a 64-token overlap: the first chunk plus up to 15 chunks from the end.

1. **Adapt.** The Phase-1 backbone with its method, an attention-pooling layer and one output logit trains on one chunk per case per step: the last chunk half the time, otherwise a random one, labelled with the case outcome. Each run has a time budget (60–80 min) and keeps the checkpoint with the best validation AUC.
2. **Encode.** The adapted backbone turns every chunk of the held-out cases, dev and test into a vector.
3. **Read the whole case.** Seven small readers over the chunk vectors and positions (last chunk, first chunk, mean, max, attention, GRU + attention, transformer) and a section study (first / middle / last third, last 4 chunks). They train on the 5,382 held-out cases, which the backbone never saw. The reader with the best AUC on the 949 validation cases is used.
4. **Calibrate.** A decision threshold is fitted on dev and kept only if fitting on one half of dev and scoring the other half (both ways) beats threshold 0.

**Students (S5, R2):** C cut to its first 6 layers (62.4M) reads the last 4 chunks end to end. S5 learns from three signals: the gold label, the teachers' mean probability at temperature T, and where the teachers' attention reader looks across the chunks:

```
L = (1 − α)·BCE(z, y) + α·T²·BCE(z/T, p̄_T) + β·KL(a_T ‖ a_S)      α = 0.5, T = 2, β = 1
```

**Methods** (own code; W₀ = frozen Phase-1 weight, s = α/r):
- **LoRMA:** y = h + s·B·A·h, h = W₀x, with B = 0 at the start.
- **AdaLoRA:** y = W₀x + s·P·diag(E)·Q·x. Each (P column, E entry, Q row) triplet is scored by smoothed |weight × gradient| times its uncertainty, and the lowest are pruned until the rank budget falls from 12 to 8 per layer on a cubic schedule. P and Q get an orthogonality penalty (0.1).
- **BitFit:** only bias terms train (C was built with biases for this).
- **Gradual unfreezing:** the top 2, 4, 6 and then 8 of A's 16 blocks open at 0%, 20%, 40% and 60% of training, with the learning rate falling 0.85× per block downward.

**S6 ensemble:** the mean of the five member runs' logits (S1, S2, S3, S4, S5), with the same 2-fold threshold check on dev.

---

## Runs and settings

Common: 2 epochs (capped by each run's time budget), head learning rate 1e-3, gradient checkpointing, seed 13; two runs at a time, one per T4.

| Run | Notebook | Backbone | Method | Settings | Submitted |
|---|---|---|---|---|---|
| **S1** | 1 | D encoder (100.1M) | **LoRMA**, all attention + MLP layers | r 16, α 32, lr 5e-4, batch 16 × 2, 60 min | yes |
| S2 | 1 | C (112.8M) | BitFit | lr 3e-3, batch 16 × 2, 60 min | yes |
| S5 | 1 | C first 6 layers (62.4M) | **Knowledge distillation** from S1 + S2 (212.9M) | lr 5e-5, batch 4 × 4, 75 min, α 0.5, T 2, β 1 | yes |
| R1 | 1 | D encoder | Full fine-tuning (reference for S1) | lr 2e-5, 60 min | yes |
| R2 | 1 | same student as S5 | Gold labels only (control for S5) | lr 5e-5, 75 min | yes |
| S3 | 2 | B (121.8M) | AdaLoRA, rank 12 → 8 | α 24, lr 1e-3, orth 0.1, batch 8 × 4, 70 min | not scored (cap) |
| S4 | 2 | A (162.7M) | Gradual unfreezing, top 2 → 8 of 16 blocks | lr 5e-5, decay 0.85, batch 8 × 4, 80 min | not scored (cap) |
| R3 | 2 | B | LoRMA (reference for S3) | r 16, α 32, lr 5e-4, 70 min | not scored (cap) |
| TF-IDF | 2 | none | Word 1–2-gram TF-IDF + logistic regression (CPU) | no pretraining reference | dev only |
| **S6** | `build_s6.py` | S1 + S2 + S3 + S4 + S5 | **Ensemble**, mean logit | dev-calibrated threshold | yes |

---

## Results

| Run | File | Trainable % of backbone | Dev macro-F1 | Dev AUC | Official macro-F1 |
|---|---|---|---|---|---|
| **S6 ensemble** | `S6_ensemble_submission.zip` | — | 0.7786 | 0.8610 | **0.7798** |
| S1 D + LoRMA | `submissionS1.zip` | 2.65 | 0.7729 | 0.8540 | 0.7693 |
| S5 KD student | `submissionS5.zip` | 100 | 0.7742 | 0.8565 | 0.7607 |
| R2 student, no KD | `submissionR2.zip` | 100 | 0.7383 | 0.8171 | 0.7341 |
| S2 C + BitFit | `submissionS2.zip` | 0.07 | 0.7308 | 0.8077 | 0.7336 |
| R1 D full fine-tuning | `submissionR1.zip` | 100 | 0.7219 | 0.8001 | 0.7243 |
| R3 B + LoRMA | `submissionR3.zip` | 3.47 | 0.7524 | 0.8318 | — |
| S4 A + gradual unfreezing | `submissionS4.zip` | 41.19 | 0.7118 | 0.7887 | — |
| S3 B + AdaLoRA | `submissionS3.zip` | 2.62 | 0.6747 | 0.7336 | — |
| TF-IDF + LR | — | — | 0.6767 | 0.7567 | — |

S3, S4 and R3 have no official score because Codabench's daily submission cap was reached; S3 and S4 are still S6 ensemble members.

Distillation: the student has 29% of the teachers' parameters (62.4M vs 212.9M) and runs about 10× faster (44.9 vs 4.28 dev cases/s).

**Best model:** the S6 ensemble. Best single model: S1, D encoder + LoRMA.

---

## How to run

1. **Notebook 1.**
   - Inputs: the tokenizer dataset (`t2_unigram.model`) and the C and D checkpoint datasets.
   - Secret `HF`; GPU T4 × 2; internet on. **Save & Run All**: ~4.5–5 h (S1 | S2, then R1 | S5, then R2).
   - Download `p4_nb1/ens_nb1.json`.
2. **Notebook 2.**
   - Inputs: the tokenizer dataset (`t1_bpe.json`) and the A and B checkpoint datasets.
   - Same settings. ~4 h (S3 | S4, then R3; TF-IDF on CPU alongside).
   - Download `p4_nb2/ens_nb2.json`.
3. **S6 ensemble** (CPU), with `cj_lib.py` from either notebook's working folder:
   ```
   python build_s6.py ens_nb1.json ens_nb2.json --lib <folder with cj_lib.py> --out p4_s6
   ```
   It writes `p4_s6/S6/submission.zip`.

Each notebook's results cell also prints:
- the readers and the section study per run;
- attention share by case position;
- errors by case length;
- the distillation comparison (notebook 1);
- the AdaLoRA rank allocation per block (notebook 2).

---

## Reproducibility

Re-run the notebooks to obtain results; scores will differ slightly from those reported here. Seeds are fixed, but GPU computation is not deterministic (fp16 mixed precision, cuDNN kernels, atomic operations). Every run also stops on a wall-clock time budget, so the number of training steps depends on GPU speed. Kaggle's software image also changes over time.
