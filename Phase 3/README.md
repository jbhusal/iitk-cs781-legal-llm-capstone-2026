# Phase 3 — Rhetorical role prediction

**Task.** Label every sentence of a judgment with one of 13 rhetorical roles (e.g. Fact, Statute, Precedent relied upon, Ratio of the decision, Ruling by present court, None), in two domains: Corporate Law (CL) and Information Technology (IT).
**Metric.** Macro-F1 over roles, so a rare role counts as much as Fact.
**Best submission.** E1: Model D's encoder + **LoRMA** + a BiLSTM over the document's sentences, **0.5242 macro-F1** on Codabench.

---

## Files

| File | Runs | Notes |
|---|---|---|
| `fp3-01-rr-notebook1.ipynb` | E1, E2, E3 + E1_noctx (dev-only ablation) | **Contains the best model (E1); includes the run log.** Writes `legal_d.py`, `c_model.py`, `rr_lib.py`, `rr_train.py`. Re-packs checkpoint folders that Kaggle unzipped back into `.pt` files |
| `fp3-02-rr-notebook2.ipynb` | T_teacher, E4, E5, E6 | Includes the run log. Independent of notebook 1: downloads the data itself and trains its own distillation teacher. Writes `legal_d.py`, `a_model.py`, `b_model.py`, `c_model.py`, `rr_lib.py`, `rr_train3.py` |
| `submission1.zip` … `submission6.zip` | E1 … E6 | The six Codabench submissions |

---

## Data

| Split | CL | IT |
|---|---|---|
| Train | 40 documents, 10,593 sentences | 40 documents, 6,252 sentences |
| Dev | 5 documents | 5 documents |
| Test | 5 documents | 5 documents |

Test has 2,197 sentences. Document ids are unique across the two domains, so one prediction file holds both. PrecedentOverruled has only 3 training sentences, and Dissent and PrecedentOverruled never occur in dev.

---

## Method (shared by every run)

1. **Sentence encoder.** The Phase-1 backbone (frozen, plus the run's adapter) reads each sentence, cut at 128 tokens, and its token states are mean-pooled. A and B read only left to right, so each sentence is fed twice and the second copy is pooled; every pooled token then sees the whole sentence.
2. **Extra features.** Each sentence vector gets its relative position in the document (0–1) and a CL/IT domain flag, then a projection layer.
3. **Document context.** A 2-layer BiLSTM (hidden 256) runs over the document's sentences. Long documents are split into windows of 64 sentences with stride 48, and overlapping logits are averaged.
4. **Loss.** Cross-entropy weighted by 1/√(label frequency); the distillation run adds the teacher term. AdamW, weight decay 0.01.
5. **Calibration.** One additive bias per class, fitted on dev by coordinate ascent to maximise macro-F1. It is kept only if a 2-fold check on dev also improves; otherwise the raw model is used.
6. **Output check.** Before zipping, the writer asserts one label per sentence, the original order, and only spellings from `label_vocab.json`.

Adapters are our own code:
- **LoRA:** y = W₀x + (α/r)BAx.
- **rsLoRA:** the same, scaled by α/√r.
- **LoRMA** (additive-identity form): h = W₀x, y = h + (α/r)B(Ah), i.e. W' = (I + (α/r)BA)W₀, with B = 0 at the start.
- **IA³:** scales the key and value outputs and the FFN down-projection input.
- **Prompt tuning:** prepends trainable embeddings.

---

## Runs and settings

Common: up to 10 epochs, early stopping with patience 3 on dev macro-F1, 35-minute budget per run, r 16, α 32, head learning rate 1e-3, seed 13; two runs at a time, one per T4.

| Run | Notebook | Backbone | Technique | Settings |
|---|---|---|---|---|
| **E1** | 1 | D encoder (~100M) | **LoRMA** | r 16, α 32, lr 5e-4; trainable 6.21M (5.84%) |
| E2 | 1 | D encoder | LoRA (same rank and layers as E1) | lr 5e-4; trainable 6.16M (5.80%) |
| E3 | 1 | C (112.8M) | IA³ | lr 3e-3; trainable 3.61M (3.10%) |
| E1_noctx | 1 | D encoder | LoRMA, **no BiLSTM context layer** | dev-only ablation |
| T_teacher | 2 | D encoder | LoRA (same setup as E2) | distillation teacher; not submitted |
| E4 | 2 | A (~163M) | rsLoRA | r 64; submitted run: α 8 (scale 1), lr 1e-4, 55-minute budget |
| E5 | 2 | B (121.8M) | Prompt tuning | 20 soft prompt vectors, lr 3e-3 |
| E6 | 2 | C (112.8M, full fine-tuning) | **Knowledge distillation** from T_teacher | 0.5 × KL(teacher, T = 2) + 0.5 × CE(gold), lr 3e-5 |

E4 note: the settings cell in `fp3-02` holds the first E4 configuration (α 16, lr 5e-4, 35 min), which reached only 3 epochs on the slow A backbone and collapsed to two labels. To reproduce the submitted E4, set `alpha=8, lr=1e-4` and a 55-minute budget for E4.

---

## Results

Dev macro-F1 is the best epoch on the dev set. The "bias" column is the 2-fold-checked calibrated score.

| Run | Dev macro-F1 (raw) | Dev macro-F1 (bias, 2-fold) | Official macro-F1 |
|---|---|---|---|
| **E1 D + LoRMA** | **0.5749** | 0.5351 | **0.5242** |
| E6 C student (KD) | 0.5381 | 0.5051 | 0.5126 |
| E3 C + IA³ | 0.5473 | 0.4838 | 0.4985 |
| E2 D + LoRA | 0.5159 | 0.4551 | 0.4805 |
| E4 A + rsLoRA | — (submitted run) | — | 0.4420 |
| E5 B + prompt tuning | 0.4239 | 0.3825 | 0.4206 |
| E1_noctx (ablation) | 0.4109 | 0.3842 | not submitted |

Submission files `submission1.zip` … `submission6.zip` correspond to E1 … E6.

Distillation: the teacher (D + LoRA) scored 0.5159 dev at 233 sentences/s; the student (C) scored 0.5381 at 259 sentences/s.

**Best model:** E1, D encoder + LoRMA, from notebook 1.

---

## How to run

1. Inputs: the tokenizer dataset (`t1_bpe.json`, `t2_unigram.model`) and the Phase-1 checkpoint datasets (notebook 1: C and D; notebook 2: A, B, C and D).
2. Secret `HF` (the notebooks download the Phase-3 data from Hugging Face). Accelerator **GPU T4 × 2**, internet on.
3. **Save Version → Save & Run All.** Each notebook takes ~80–90 min.
   - Notebook 1 runs E1 | E2, then E3 | E1_noctx.
   - Notebook 2 runs T_teacher | E4, then E6 (after the teacher) | E5.
4. Outputs:
   - notebook 1: `p3_nb1/E1|E2|E3/submission.zip`, plus `runs.csv`, per-run `metrics.json` and `dev_confusion.csv`;
   - notebook 2: `p3_nb2/E4|E5|E6/submission.zip`.

---

## Reproducibility

Re-run the notebooks to obtain results; scores will differ slightly from those reported here. Seeds are fixed, but GPU computation is not deterministic (fp16 mixed precision, cuDNN kernels, atomic operations). Every run also stops on a wall-clock time budget, so the number of training steps depends on GPU speed. Kaggle's software image also changes over time.
