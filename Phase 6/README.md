# Phase 6 — Abstractive summarisation

**Task.** Write an abstractive summary of a full legal judgment.
**Metric.** BERTScore F1 (roberta-large).
**Best submission.** S1, D encoder-decoder + LoRMA with a plan-then-write pipeline: **0.8528 BERTScore F1** on Codabench.

---

## Files

| File | What it is |
|---|---|
| `p6_summ_all_runs.ipynb` | One notebook with every run (S1–S7 plus two no-model references). Writes `legal_d.py`, `c_model.py`, `a_model.py`, `b_model.py`, `sum_lib.py` (segmentation, alignment, length plan, methods, generation, scoring, self-tests) and `sum_train.py` (one run per process) |
| `submission1.zip` … `submission7.zip` | The seven Codabench submissions, one per run: `submission1.zip` = S1, …, `submission7.zip` = S7 |

---

## Data

- **Train:** 7,030 judgments with reference summaries. 27 duplicates and 2 empty documents are dropped, and 40 documents are held out to tune the length knob and 80 for dev scores, leaving 6,881 for training.
- **Test:** 100 judgments.
- **Length:** judgments are long, median 3,128 words (90th percentile 7,843; longest 139,645). Summaries are long too, median 640 words (90th percentile 1,503); median compression 0.218.
- **Scoring cap:** BERTScore reads only the first ~510 tokens of each summary, so the opening of the summary carries the score.

---

## Method: plan, then write section by section

No Phase-1 model reads more than ~1,024 tokens, and no single pass can write a 640-word summary. So every run plans first, then writes part by part. Runs differ only in backbone and technique.

1. **Segment** each judgment into ~450-word parts at sentence boundaries (dev: 10.1 parts per judgment).
2. **Align** every reference-summary sentence to the part that supports it best, using IDF-weighted word overlap and a near-monotone dynamic programme. One long pair becomes many short (part, sub-summary) pairs that fit the models: divide-and-conquer training (Gidiotis & Tsoumakas 2020). This gave 202,430 aligned sentences (median support 0.89) and 52,020 training pairs (median sub-summary 93 words).
3. **Plan** the length. Total words come from a power law fitted on train, words = 4.58 × judgment_words^0.61, clipped to 232–2,016. The words are shared across parts by a learned positional profile. Parts with fewer than 12 words of budget are skipped, and their words are re-shared to the other parts.
4. **Write** each planned part with the adapted model, conditioned on the cue `[<position> part] Summary (about N words):`, which controls length and position (Fan et al. 2018). The parts are then joined in order.
5. **Tune** one knob on the 40 tune documents: the length scale (0.8, 1.0 or 1.25 × plan), chosen by BERTScore.

**Generation:** batched greedy decoding with a KV cache, repetition penalty and n-gram blocking; ids the tokenizer cannot decode, and special tokens, are banned. Self-tests check that cached generation equals the full forward pass and that batched equals one-at-a-time.

**S7 consensus:** for every planned part, keep the version (out of S1–S6) that agrees most with the others. This is segment-level MBR with ROUGE-1.

**Distillation (S6):** the student is trained on the gold words plus the teacher's word distributions (T 2, weight 0.5), following Kim & Rush (2016).

**Methods** (own code):
- **LoRMA** on encoder, decoder and cross-attention.
- **LoRA.**
- **QLoRA:** own NF4 with double quantisation, plus an fp16 weight cache at generation.
- **Houlsby adapters.**
- **Prefix tuning:** 32 learned states at every encoder layer, from real activations.

---

## Runs and settings

Common: 2 epochs (capped by each run's budget), seed 13, 400 validation pairs, gradient checkpointing.

| Run | Backbone | Technique | Settings |
|---|---|---|---|
| **S1** | D encoder-decoder | **LoRMA**, r 16, α 32 (encoder, decoder, cross-attention) | lr 5e-4, batch 8 × 2, 55 min; KD teacher |
| S2 | C prefix-LM | Houlsby adapters, bottleneck 64 | lr 1e-3, batch 8 × 2, 55 min |
| S3 | B | QLoRA: NF4 4-bit + LoRA r 16 | lr 5e-4, batch 6 × 3, 60 min |
| S4 | A | LoRA, r 16 | lr 5e-4, batch 4 × 4, 60 min |
| S5 | D encoder-decoder | Prefix tuning, 32 states per encoder layer | lr 3e-3, batch 8 × 2, 50 min |
| S6 | half of D: encoder + decoder layers 0, 2, …, 10 (109.5M) | **Knowledge distillation** from S1, α 0.5, T 2 (full fine-tuning) | lr 1e-4, batch 8 × 2, 55 min |
| S7 | S1–S6 | Segment-level consensus (MBR, ROUGE-1) | CPU |
| Lead / Extract | none | First N words / most central sentences of every planned part | dev-only references |

---

## Results

Dev scores are on the 80 held-out judgments. Official scores are on the 100 test judgments.

| Run | File | Trainable | Storage (MB) | Dev BERTScore F1 | Dev ROUGE-1 | Dev ROUGE-L | Official BERTScore F1 |
|---|---|---|---|---|---|---|---|
| **S1 D + LoRMA** | `submission1.zip` | 6.09M (3.13%) | 383 | 0.8557 | 0.5737 | 0.3034 | **0.8528** |
| S7 consensus | `submission7.zip` | — | — | 0.8562 | 0.5731 | 0.3044 | 0.8524 |
| S6 KD student | `submission6.zip` | 100% | 209 | 0.8542 | 0.5545 | 0.2955 | 0.8515 |
| S2 C + adapters | `submission2.zip` | 2.80M (2.48%) | 220 | 0.8539 | 0.5690 | 0.2943 | 0.8509 |
| S3 B + QLoRA | `submission3.zip` | 4.25M (3.49%) | 97 | 0.8466 | 0.5303 | 0.2680 | 0.8452 |
| S5 D + prefix | `submission5.zip` | 0.29M (0.15%) | 371 | 0.8407 | 0.4028 | 0.2137 | 0.8366 |
| S4 A + LoRA | `submission4.zip` | 4.00M (2.46%) | 318 | 0.8301 | 0.4595 | 0.2279 | 0.8260 |
| Extract (no model) | — | — | — | 0.8511 | 0.5538 | 0.2960 | not allowed |
| Lead (no model) | — | — | — | 0.8341 | 0.5089 | 0.2516 | not allowed |

Distillation: the student has 56% of D's parameters and writes 80 dev documents per minute against S1's 27. A uniform-budget ablation of S1 (no positional profile) scored 0.8551 dev BERTScore.

**Best model:** S1, D encoder-decoder + LoRMA.

---

## How to run

1. Inputs: the tokenizer dataset (`t1_bpe.json`, `t2_unigram.model`) and the A, B, C and D checkpoint datasets.
2. Secret `HF`; **GPU T4 × 2**; internet on. Internet also fetches `bert-score` / roberta-large, used for local scoring only, never to train or to write summaries.
3. **Save Version → Save & Run All.** The real run took 4.8 h. Two jobs run at a time, and S6 waits for its teacher S1.
4. Outputs: `p6_out/S1 … S7/submission.zip` and `p6_out/report_files.zip`.

The results cell also prints, per run:
- words, repeated 3-grams, novel 4-grams and numbers found in the source;
- the length scale chosen;
- training and generation time;
- the distillation and QLoRA comparisons;
- an example summary.

---

## Reproducibility

Re-run the notebooks to obtain results; scores will differ slightly from those reported here. Seeds are fixed, but GPU computation is not deterministic (fp16 mixed precision, cuDNN kernels, atomic operations). Every run also stops on a wall-clock time budget, so the number of training steps depends on GPU speed. Kaggle's software image also changes over time.
