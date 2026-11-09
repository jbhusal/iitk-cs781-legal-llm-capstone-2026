# Phase 1 — Legal language model pretraining

**Task.** Given a 512-word prompt from an Indian legal document, write the next 128 words.
**Metric.** BERTScore F1 (roberta-large), with ROUGE-L and BLEU-4 as secondary metrics.
**Best submission.** Model D (encoder-decoder, 194.5M) with MBR decoding: **0.8389 BERTScore F1** on Codabench (`modelD/submission.zip`).

Four models from four architecture families were trained from random initialisation on the official Phase-1 corpus only. Their checkpoints are the starting point for every later phase.

---

## Files

| Path | What it does |
|---|---|
| `p1_00_data_prep.ipynb` | Cleans the four raw training sources (Acts, High Courts, Supreme Court, Tribunals) into `clean_<source>.txt`. Builds a local test-shaped dev set (`dev_prompts.json`, `dev_truth.json`: 512-word prompt → true 128-word continuation) |
| `p1_01_tokenizer.ipynb` | Trains the two tokenizers: **T1** byte-level BPE (32k, used by A and B) and **T2** SentencePiece Unigram (32k, used by C and D). Special tokens are reserved for later phases |
| `p1_02_bench.ipynb` | Kernel and throughput benchmark on the T4 (attention backends, shared building blocks) before committing GPU hours; trains no model |
| `modelA/p1-03-model-a.ipynb` | Trains Model A (resumable); includes the run log |
| `modelA/p1-03-1-predict-a.ipynb` | Model A test predictions: tries output fixes on 100 dev prompts, keeps the best; includes the run log |
| `modelA/submission.zip` | Model A submission (**0.8231**) |
| `modelB/p1_04_model_B.ipynb` | Trains Model B (resumable) |
| `modelB/p1_4.1_predict_B.ipynb` | Model B test predictions (output settings compared on 100 dev prompts) |
| `modelB/submission.zip` | Model B submission (**0.8326**) |
| `modelC/p1-05-model-c.ipynb` | Trains Model C (resumable); includes the run log |
| `modelC/p1_5.2_predict_C.ipynb` | Model C test predictions: repetition penalty 1.3 / 1.4 / 1.5 on 300 dev prompts. 1.3 is best, the setting used in the submission |
| `modelC/submission.zip` | Model C submission (**0.8362**) |
| `modelD/p1-06-model-D.ipynb` | Trains Model D on both GPUs (DDP); writes `legal_d.py` (model, trainer, generation, scoring); includes the run log |
| `modelD/p1_6.1_predict_D.ipynb` | Model D test predictions: compares decoding settings and MBR on 300 dev prompts, writes `submission.zip` |
| `modelD/p1_6.2_model_D_cont.ipynb` | Continues D's training for 4 h with a fresh learning-rate schedule (D-cont) |
| `modelD/submission.zip` | **Model D + MBR submission (0.8389), the best** |

To reproduce the best model, run in order: `p1_00_data_prep` → `p1_01_tokenizer` → `modelD/p1-06-model-D` → `modelD/p1_6.1_predict_D`.

---

## Data

- **Training text:** the four official Phase-1 sources, cleaned, streamed round-robin across sources and separated by an end-of-document token. Byte positions are saved in every checkpoint, so a resumed run never repeats text.
- **Dev set:** there is no public dev set, so `p1_00_data_prep` builds one from the training sources in the exact test format.
- **Output format:** `prediction.json` at the zip root, all 1,000 official test ids, the continuation only (no prompt), at most 128 words. C and D always reach exactly 128 words; a few A and B outputs stop earlier (A 25, B 11 of 1,000). Every submission is checked automatically before zipping.

---

## Model architectures

| | A | B | C | D |
|---|---|---|---|---|
| Family | Decoder-only | Efficient decoder | Prefix-LM (hybrid encoder used as an LM) | Encoder-decoder |
| Parameters | ~163M | 121.8M | 112.8M | 194.5M |
| Shape | 16 × 896 | 24 × 640 (deep, narrow) | 14 × 768 | 12 + 12 × 768 |
| How the prompt is read | left to right | left to right, 512-token window (+ global layers) | both directions | both directions, in a separate encoder |
| How the continuation is written | same stack | same stack | same stack, causal | separate decoder + cross-attention |
| Attention | GQA 14q / 2kv, RoPE | MQA (1 kv head), QK-norm; 5 of every 6 layers local (512), every 6th global with no position encoding | GQA 12q / 4kv, QK-norm, prefix mask | GQA 12q / 4kv, QK-norm; RoPE in self-attention, none in cross-attention |
| MLP / norm | SwiGLU, RMSNorm, no biases | SwiGLU, RMSNorm | GELU with biases, RMSNorm | SwiGLU, RMSNorm, no biases |
| Tokenizer | T1 (BPE) | T1 (BPE) | T2 (Unigram) | T2 (Unigram) |
| Training objective | next token | next token | 50% test-shaped windows (loss on the continuation) + 50% next token | 100% test-shaped windows (prompt 512–768 → continuation 256–512 tokens) |
| Optimiser / GPUs | AdamW, 1 GPU | AdamW, DataParallel | AdamW, DataParallel | Muon + AdamW, DDP |

**Why these four:**
- **A** is the standard modern baseline and the BPE submission.
- **B** tests efficiency: most attention in legal text is local, and one shared key/value head cuts memory.
- **C** lets every prompt token see the whole prompt, matching the test task.
- **D** gives each prompt token only to the encoder and each continuation token only to the decoder, so it has ~1.7× C's parameters at ~15% less compute per token. Its encoder is also the backbone for Phases 2–5 and 7.

---

## Training (Model D, the best model)

- **Data per step:** 61,440 tokens. Window shapes (prompt / continuation tokens): 768 / 256 (40%), 640 / 384 (30%), 512 / 512 (30%).
- **Optimiser:** Muon (5 Newton–Schulz steps) for hidden weight matrices, AdamW for embeddings and norms; fp16 autocast with GradScaler (the T4 has no bf16).
- **Schedule:** warmup–stable–decay on **wall-clock time** (300 warmup steps, peak 8e-4, 1-√ decay over the last 20%), so the learning rate finishes decaying exactly when the time budget ends.
- **Efficiency:** one process per GPU (DDP via `torch.distributed.run`). The 32k-word output layer runs only on continuation tokens, in chunks. Flat checkpoints with resume, pushed to a Kaggle dataset.
- **Run:** 569.5 min on 2× T4, 10,585 steps, **650M tokens** at ~19k tokens/s. Final checkpoint `D_step10585_tokens650342400.pt`, dev BERTScore 0.8400 (300 prompts).

D-cont (`p1_6.2_model_D_cont`) resumed this checkpoint and trained 4 h more: re-warm to 4e-4 over 300 steps, hold, then decay over the last 30%, for 932M tokens in total. Its dev loss improved, but its dev BERTScore did not (0.8384), so the 0.8389 model is the one used in later phases.

---

## Decoding

The setting used for the final submissions: greedy decoding with **repetition penalty** (1.2 for A and B, 1.3 for C and D) on generated tokens, **no repeated 3-gram**, special tokens banned, generation until each row passes 128 words or reaches the token limit, then cut to 128 words.

**MBR** (`p1_6.1_predict_D`): for each prompt, 1 greedy + 7 sampled continuations (temperature 0.8, top-p 0.9). The candidate with the highest mean ROUGE-L to the others is kept. MBR is used for the test set only if it beats the best greedy setting on dev by more than 0.001. It did: 0.8411 vs 0.8400.

---

## Results

| Model | Training text | Dev BERTScore | Submission file | Official BERTScore F1 |
|---|---|---|---|---|
| A | 52M tokens | 0.8245 (100 prompts) | `modelA/submission.zip` | 0.8231 |
| B | ~4 h run | 0.8393 (100 prompts) | `modelB/submission.zip` | 0.8326 |
| C | 448M tokens | 0.8376 (300 prompts) | `modelC/submission.zip` | 0.8362 |
| **D + MBR** | 650M tokens | 0.8411 (300 prompts) | `modelD/submission.zip` | **0.8389** |
| D-cont | 932M tokens | 0.8384 (300 prompts) | — | — |

Bits-per-byte on the dev continuations (lower is better): C 0.7533, D 0.7146, D-cont 0.6966.

---

## How to run (best model)

1. **`p1_00_data_prep`**: secret `HF`, internet on. Save its output as a Kaggle dataset (used by the other notebooks as `legallm-phase1-data`).
2. **`p1_01_tokenizer`**: attach the data dataset; secret `KAGGLE_API_TOKEN` publishes `t1_bpe.json` and `t2_unigram.model` as a dataset (`legallm-tokenizers`).
3. **`modelD/p1-06-model-D`**:
   - attach the data and tokenizer datasets;
   - secret `KAGGLE_API_TOKEN` (pushes checkpoints to a dataset such as `legallm-checkpoints-d`);
   - GPU T4 × 2; ~9.5 h of training, resumable across sessions.
4. **`modelD/p1_6.1_predict_D`**:
   - attach data, tokenizers and the checkpoint dataset;
   - secret `HF`, GPU on;
   - writes `phase1_predict_d/submission.zip`. MBR adds ~26 min for 1,000 prompts.
   - The notebook loads the **newest** `D_step*` checkpoint it finds. To reproduce the submitted model, attach the dataset version holding `D_step10585_tokens650342400.pt`, not the D-cont one.

The A, B and C trainers and predictors follow the same pattern: attach the data and tokenizer datasets, plus the model's own checkpoint dataset when resuming or predicting, then Save & Run All. `modelA/p1-03-model-a.ipynb` points to fixed dataset paths (`/kaggle/input/datasets/<user>/…`); edit those to your own.

---

## Reproducibility

Re-run the notebooks to obtain results; scores will differ slightly from those reported here. Seeds are fixed, but GPU computation is not deterministic (fp16 mixed precision, cuDNN kernels, atomic operations). Many runs also stop on a wall-clock time budget, so the number of training steps depends on GPU speed. Kaggle's software image also changes over time.
