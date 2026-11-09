# Phase 7 — Prior case retrieval (PCR)

**Task.** For each query judgment, rank its split's candidate pool and return the 7 prior cases it most likely cites.
**Metric.** Micro-F1@7 (secondary: micro-precision@7, micro-recall@7).
**Best submission.** S7, an ensemble of six dual-encoder retrievers: **0.3479 micro-F1@7** on Codabench. Best single model: S6, the distilled half-depth D student (0.3379).

---

## Files

| File | What it is |
|---|---|
| `p7_pcr_all_runs.ipynb` | One notebook with every run (S1–S7, R1, BM25 reference). Writes `legal_d.py`, `c_model.py`, `a_model.py`, `b_model.py`, `pcr_lib.py` (data, BM25, alignment, encoders, methods, training, scoring, self-tests) and `pcr_train.py` (one run per process) |
| `submission1R1.zip` … `submission8S7.zip` | The eight Codabench submissions; the number is the submit order and the suffix is the run |

---

## Data

`Phase_7_PCR/{Train,Dev,Test}/{split}.json` with `{split}_candidates.json`. Each split's queries are ranked only against that split's own candidate pool.

- **Query text:** a list of sentences.
- **Labels:** train and dev queries carry `relevant_candidates`; test queries do not.
- **Ids:** IndianKanoon case ids. Internal ids are `<split>:q|c:<id>`, so overlapping ids across splits never collide.
- **Robustness:** cited ids written without zero-padding are matched to the pool by number, and a query that also appears in its own pool is never retrieved for itself.

The notebook prints, per split: query and pool sizes, words per document, cited cases per query, and cited ids missing from the pool.

---

## Method (same for every run)

1. **Chunk.** Every judgment is cut into 360-word chunks (at most 12, spread evenly, first and last always kept). Chunks are capped at 512 tokens.
2. **BM25** (own code, k1 1.2, b 0.75, query term weight log(1 + tf)) over each split's pool. It gives the **hard negatives** (top-10 BM25 candidates that are not cited) and a lexical score for fusion.
3. **Evidence-aligned pairs.** For each (query, cited case), the chunk pairs with the highest TF-IDF cosine. The model trains on the passages that make one judgment cite the other, not on random windows.
4. **Dual encoder.** One Phase-1 backbone for both sides, plus attention pooling, gives one L2-normalised vector per chunk. The loss is **InfoNCE** (temperature 0.05) over in-batch candidates + 1 BM25 hard negative per query. Other cases the same query cites are masked out, so they never count as negatives. 10% of training queries are held out for validation loss, and the best checkpoint is kept.
5. **Document score** from cached chunk vectors, three ways:
   - **mean:** one vector per judgment;
   - **max-sim:** each query chunk takes its best candidate chunk, then average (late interaction at chunk level);
   - **max-pair.**
6. **Fuse.** final = z(model score) + w·z(BM25), w ∈ {0, 0.25, 0.5, 0.75, 1}. The aggregation and w are chosen by an honest **2-fold check on dev queries**: choose on one half, score the other, both ways.

**S1 zero-shot check:** before training, the untrained Phase-1 D encoder is scored on dev, to measure how much Phase-1 representations transfer.

**Distillation (S6):** listwise. The student (D encoder layers 0, 2, …, 10) matches the teacher's (S1) softmax over the in-batch similarity matrix: KL at T 2, weight 0.5 against InfoNCE.

**S7 ensemble:** the mean of the z-scored final scores of S1–S6. R1 is excluded.

**Methods** (own code):
- **LoRMA** on every attention + MLP layer.
- **IA³:** scales on keys, values and the MLP input.
- **QLoRA:** own NF4 with double quantisation, plus LoRA.
- **Partial fine-tuning:** top 4 blocks + final norm.
- **Prompt tuning:** 20 learned input vectors, started from real token embeddings.

---

## Runs and settings

Common: seed 13, gradient checkpointing, 1,500 steps planned (capped by each run's budget), 12 validation batches.

| Run | Backbone | Technique | Settings |
|---|---|---|---|
| S1 | D encoder (100.1M) | **LoRMA**, r 16, α 32 | lr 5e-4, 12 queries / step, 35 min; KD teacher; zero-shot check |
| S2 | C (112.8M) | **IA³** | lr 3e-3, 12 queries / step, 35 min |
| S3 | B (121.8M) | **QLoRA** (NF4 + LoRA r 16) | lr 5e-4, 8 queries / step, 40 min |
| S4 | A (162.7M) | **Partial / layer-wise fine-tuning**, top 4 blocks | lr 1e-4, 8 queries / step, 40 min |
| S5 | D encoder | **Prompt tuning**, 20 vectors | lr 1e-2, 12 queries / step, 30 min |
| S6 | half of D (62.3M) | **Knowledge distillation** from S1 + full fine-tuning | lr 1e-4, 12 queries / step, 35 min, α 0.5, T 2 |
| S7 | S1–S6 | **Ensemble**, mean of z-scores | CPU |
| R1 | D encoder | Ablation: S1 **without** BM25 hard negatives (in-batch only) | as S1, 30 min |

---

## Results

| Run | Setup | File | Official micro-F1@7 |
|---|---|---|---|
| **S7** | Ensemble of S1–S6 | `submission8S7.zip` | **0.3479** |
| S6 | KD student, half of D | `submission7S6.zip` | 0.3379 |
| S1 | D encoder + LoRMA | `submission2S1.zip` | 0.3349 |
| S2 | C + IA³ | `submission3S2.zip` | 0.3337 |
| S3 | B + QLoRA | `submission4S3.zip` | 0.3136 |
| S5 | D encoder + prompt tuning | `submission6S5.zip` | 0.3107 |
| R1 | S1 without hard negatives | `submission1R1.zip` | 0.3054 |
| S4 | A + partial fine-tuning | `submission5S4.zip` | 0.2865 |

The notebook's results cell prints the matching dev numbers for every run:
- micro-F1@7 (2-fold and fit), precision and recall;
- the model-alone score per aggregation, and the BM25-weight grid;
- BM25 alone, and the zero-shot vs trained comparison;
- the hard-negative ablation and the distillation trade-off (parameters, documents/s);
- failure analysis by number of cited cases and by query length, and the rank of cited cases.

**Best model:** the S7 ensemble. Best single model: S6, the distilled student, which also beats its teacher S1.

---

## How to run

1. Inputs: the tokenizer dataset (`t1_bpe.json`, `t2_unigram.model`) and **all four** A, B, C and D checkpoint datasets. The first cell stops with an error if a checkpoint is missing.
2. Secret `HF`; **GPU T4 × 2**; internet on.
3. **Save Version → Save & Run All.** About 4.5–5 h. Two jobs run at a time, and S6 waits for its teacher S1. The scheduler skips any run that could not finish before 11 h.
4. Outputs: `p7_out/<run>/submission.zip` and `p7_out/report_files.zip`. Each `prediction.json` maps every test query id to exactly 7 unique candidate ids from its pool, ordered by relevance; the last cell checks this for every file.

---

## Reproducibility

Re-run the notebooks to obtain results; scores will differ slightly from those reported here. Seeds are fixed, but GPU computation is not deterministic (fp16 mixed precision, cuDNN kernels, atomic operations). Every run also stops on a wall-clock time budget, so the number of training steps depends on GPU speed. Kaggle's software image also changes over time.
