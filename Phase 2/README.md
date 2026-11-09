# Phase 2 — Legal named entity recognition (L-NER)

**Task.** Find every entity in a legal judgment as an exact character span with one of 12 labels: APP, RESP, A.COUNSEL, R.COUNSEL, JUDGE, WIT, AUTH, COURT, STAT, PREC, DATE, CASENO.
**Metric.** Strict macro-F1: a prediction counts only if its start offset, end offset and label all match exactly.
**Best official score.** **0.5664** strict macro-F1. **Best model on validation:** Model C + LoRMA (`p2_03`).

---

## Files

| Notebook | Run | What it does |
|---|---|---|
| `p2_02_S1_modelD_ner.ipynb` | S1 | Model D's encoder + tagging head, **full fine-tuning**, 5-fold |
| `p2_03_S2_modelC_lorma.ipynb` | S2 | Model C (prefix-LM, window read in both directions) + **LoRMA** adapters + tagging head, 5-fold. Best model on validation |
| `p2_04_experiments_any_model_method.ipynb` | comparison runs | One notebook for the backbone × method grid: set `BACKBONE` = `"C"` or `"D"` and `METHOD` = `"lorma"`, `"lora"` or `"full"` in the first code cell. Each setting saves to `p2_runs/<BACKBONE>_<METHOD>/` |
| `sub-1.zip` … `sub-4.zip` | — | The four Codabench submissions (scores below) |

All three share the same data loading, tokenisation, tagging scheme, decoding and scoring code, so differences come only from the backbone and the adaptation method.

---

## Data

- **Train:** folds 1 and 2 (70 annotated judgments), each mapping a document id to its full text and character-level gold spans.
- **Test:** fold 3, text only.
- **Validation:** there is no public dev split, so the 70 training documents are split into **5 folds** (fixed shuffle, seed 0). Every training document is predicted by a model that never saw it (out-of-fold).

---

## Method

1. **Word pieces with exact offsets.** A regex splits the text into words, numbers, underscores and single punctuation marks, keeping each piece's character start and end. Each piece is tokenised with the T2 SentencePiece tokenizer. For pieces attached to the previous one (punctuation, footnote digits), the word-start marker is removed so tokens line up with the text.
2. **BIO tags.** Gold spans become B-/I- tags on the first token of each word piece (25 tags = O + B/I × 12 labels). Other sub-tokens are ignored in the loss.
3. **Windows.** Documents are cut into 512-token windows with stride 256. In training, each epoch uses a random window offset, which acts as free augmentation. At prediction time, overlapping windows are averaged with a triangular weight, so the centre of a window counts most.
4. **Encoder.**
   - **D:** the encoder only; the decoder is deleted.
   - **C:** the whole real window is the bidirectional prefix, so every token sees the full window, and padding is isolated.
   - In both cases a dropout + linear tagging head sits on top.
5. **Loss.** Cross-entropy with class weights (count(O) / count(tag))^0.25, clipped to [1, 6], so rare entity tags weigh more.
6. **Decoding.** A **constrained Viterbi** pass forbids invalid transitions (I-X may only follow B-X or I-X), then tags become character spans.
7. **Per-label bias.** One additive log-bonus per label, chosen on the out-of-fold predictions by coordinate search on a coarse grid (−1 … +2). This raises macro-F1 directly.
8. **Test predictions.** The 5 fold models' probabilities are averaged, then decoded twice: `submission_raw.zip` (no bias) and `submission_tuned.zip` (with the tuned bias).

**LoRMA** (own code): W' = (I + (α/r)·B·A)·W₀, applied to the layer output: z = W₀x, y = z + (α/r)·B(Az) + bias. B starts at 0, so training begins exactly at the Phase-1 model. Adapters go on q, k, v, o, up and down projections; the backbone stays frozen.

---

## Runs and settings

Common to all runs: window 512, stride 256, batch 8 × accumulation 2 (16 windows), AdamW, weight decay 0.01, 6% warmup then cosine decay, dropout 0.1, class weights on, seed 0 + fold.

| Run | Backbone | Method | Trainable | Epochs | Learning rates | Fold-0 dev strict macro-F1 |
|---|---|---|---|---|---|---|
| S1 (`p2_02`) | D encoder (~100M) | Full fine-tuning | all weights + head | 8 | backbone 5e-5, head 1e-3 | 0.424 |
| **S2 (`p2_03`)** | C (112.8M) | **LoRMA**, r 16, α 32 | adapters + head (backbone frozen) | 10 | adapters 5e-4, head 1e-3 | **0.545** |
| `p2_04` | C or D | LoRMA / LoRA (r 16, α 32) / full | depends on method | 8 | adapters 5e-4, backbone 5e-5 (full), head 1e-3 | — |

Fold-0 scores are on the same 14 held-out documents. The full 5-fold out-of-fold score, per-label F1 and an error breakdown are printed at the end of each notebook.

---

## Submissions

| File | Official strict macro-F1 | Entities predicted (35 test documents) |
|---|---|---|
| `sub-1.zip` | **0.5664** | 3,574 |
| `sub-2.zip` | 0.5358 | 3,288 |
| `sub-3.zip` | 0.4724 | 5,355 |
| `sub-4.zip` | 0.3814 | 4,994 |

Each submission is a 5-fold model average from the notebooks above, written as `submission_raw.zip` or `submission_tuned.zip`.

**Best model:** C + LoRMA (`p2_03`). It led on validation (fold-0 0.545 vs 0.424 for D + full fine-tuning) while training only the adapters and the head.

---

## Implementation details

- **Self-tests** (run before training, on CPU):
  - LoRMA leaves the model unchanged at step 0;
  - the adapter equals the merged weight (I + sBA)W₀;
  - padding never changes the states of real tokens.
- **Offsets:** predictions are kept only if 0 ≤ start < end ≤ document length and the label is in `labels.txt`. The writer asserts the test ids match exactly.
- **Resume:** `results.pkl` is saved after every fold; a restarted session skips finished folds.
- **Error analysis** (last cell of `p2_03`, `p2_04`): per label, the share of gold entities matched exactly, right label with wrong boundary, right span with wrong label, overlapping a different label, missed, plus spurious predictions, with boundary examples.

---

## How to run

1. Accelerator **GPU** (one T4 is enough). Internet is not needed.
2. Attach: the Phase-2 data (`fold1-text.json`, `fold1-gold.json`, `fold2-text.json`, `fold2-gold.json`, `fold3-text.json`, `labels.txt`), the tokenizer dataset (`t2_unigram.model`), and the checkpoint: `C_step13683_tokens448364544` for C, `D_step10585_tokens650342400` for D.
3. Quick check: set `FOLDS_TO_RUN=[0]` in the first code cell and run. Then set `[0, 1, 2, 3, 4]` and **Save & Run All**; finished folds are skipped.
4. Outputs in `/kaggle/working/p2_runs/<RUN_NAME>/`: `submission_raw.zip`, `submission_tuned.zip`, `results.pkl`.
5. For `p2_04`, set `BACKBONE` and `METHOD` first; the run name becomes `<BACKBONE>_<METHOD>`.

---

## Reproducibility

Re-run the notebooks to obtain results; scores will differ slightly from those reported here. Seeds are fixed, but GPU computation is not deterministic (fp16 mixed precision, cuDNN kernels, atomic operations), and Kaggle's software image changes over time.
