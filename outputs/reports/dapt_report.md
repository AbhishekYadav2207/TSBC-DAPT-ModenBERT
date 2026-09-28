# MaritimeBERT Domain-Adaptive Pretraining (DAPT) - Technical Report

**Model Identifier**: `MaritimeBERT-v1`  
**Base Model Architecture**: `answerdotai/ModernBERT-base` (149M parameters)  
**Execution Source**: `dapt/runner (1).ipynb` & `dapt/log.txt`  
**Verification Date**: September 2026  

---

## 1. Executive Summary

Domain-Adaptive Pretraining (DAPT) was executed from start to finish on maritime accident occurrence narratives from the Transportation Safety Board of Canada (TSBC). Across **984 training steps** (3 full epochs over 87,174 documents packed into 10,484 sequences of length 512), domain adaptation achieved substantial intrinsic language modeling improvements on held-out maritime narratives.

In post-training held-out evaluation against the untouched ModernBERT baseline control on the identical 585 validation sequences (299,199 tokens):
- **Held-Out MLM Loss**: Reduced from **1.5365** to **0.5247** ($\Delta = -1.0118$, **65.85% loss reduction**)
- **Held-Out Perplexity**: Reduced from **4.6484** to **1.6900** ($\Delta = -2.9584$, **63.64% perplexity reduction**)

### Core State Distinction Summary
| Model / State | Step | Held-Out MLM Loss | Perplexity | Evaluation Protocol | Persisted to Disk? | Location / Artifact Status |
| :--- | :---: | :---: | :---: | :--- | :---: | :--- |
| **`ModernBERT-base` Baseline** | 0 | 1.5365 | 4.6484 | Pre-training baseline eval | Yes | `baseline-modernbert/evaluation_metrics.json` |
| **Best validation state** | 950 | 0.4991 | 1.6473 | In-training periodic eval | **No** | Empirical minimum; not serialized (`save_steps=100`) |
| **Final training checkpoint** | 984 | 0.5151 | 1.6739 | In-training final step eval | **Yes** | `dapt/checkpoints/checkpoint-984` & `best/` |
| **Released `MaritimeBERT-v1`** | 984 | 0.5247* | 1.6900* | Post-training exported model eval | **Yes** | `dapt/outputs/experiments/MaritimeBERT-v1` |

`*` Post-training evaluation of the exported model on 585 packed validation sequences.

> [!IMPORTANT]
> **Scientific Scope Boundary**: The 63.64% perplexity reduction ($4.6484 \rightarrow 1.6900$) demonstrates valid **intrinsic held-out language modeling improvement** under the MLM self-supervised objective, establishing substantially reduced token uncertainty when predicting maritime technical terminology. It does not by itself assert downstream task superiority (classification, NER, QA, RAG) without fine-tuning adapters.

---

## 2. Environment & System Specifications

<a id="table-7"></a>
### Table 7: System Environment & Hardware Configuration
| Parameter | Value |
| :--- | :--- |
| **Operating System** | Linux-6.6.122+-x86_64-with-glibc2.39 |
| **Python Runtime** | 3.13.15 (main, Aug  6 2026, 11:06:22) [GCC 13.3.0] |
| **PyTorch Version** | 2.11.0+cu128 |
| **Transformers Version** | 5.16.1 |
| **Datasets Version** | 4.8.5 |
| **CUDA Version** | 12.8 (CUDA Available: True) |
| **GPU Accelerator** | Tesla T4 (14.56 GB VRAM) |
| **Training Duration** | 6665.35 seconds (6,665.35 s / ~1.85 hours) |
| **Average Throughput** | 2415.77 tokens/second (4.72 samples/s) |

---

## 3. Corpus Provenance & Deterministic Splitting

<a id="table-2"></a>
### Table 2: Corpus Metadata & Split Breakdown
| Partition | Documents | Percentage | Words | Characters | Artifact Path |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Train Set** | 87,174 | 90.0% | 3,446,848 | 21,921,138 | `dapt/outputs/data/train.txt` |
| **Validation Set** | 4,843 | 5.0% | 192,475 | 1,222,841 | `dapt/outputs/data/val.txt` |
| **Test Set** | 4,844 | 5.0% | 191,027 | 1,212,841 | `dapt/outputs/data/test.txt` |
| **Total Corpus** | **96,861** | **100.0%** | **3,830,350** | **24,356,820** | `outputs/maritime_corpus.txt` |

- **Cryptographic SHA-256 Hash Lock**: `852fea9a1d756a6989fd55f40646fd7e92ef7597baa4ada36d450e7e8acb6e11`
- **Split Random Seed**: `42` (deterministic random partition)
- **Exact Duplicate Document Overlap**: 0 across all pairwise splits (0 train-val, 0 train-test, 0 val-test)

---

## 4. Sequence Packing & ModernBERT Tokenization

<a id="table-3"></a>
### Table 3: Sequence Packing & Tokenizer Characteristics
| Parameter | Value | Source Artifact |
| :--- | :--- | :--- |
| **Base Tokenizer** | `answerdotai/ModernBERT-base` | `tokenizer_report.json` |
| **Vocabulary Size** | 50,368 tokens | `tokenizer_report.json` |
| **Boundary Token** | `[SEP]` (token ID 50282) | `tokenizer_report.json` |
| **Subword Fertility** | 1.6287 subwords/word | `tokenizer_report.json` |
| **Max Sequence Length** | 512 tokens | `dapt.yaml` |
| **Train Sequences Packed** | 10,484 blocks (Efficiency: **99.99%**, Waste: **0.01%**) | `dapt/log.txt` |
| **Val Sequences Packed** | 585 blocks (Efficiency: **99.89%**, Waste: **0.11%**) | `dapt/log.txt` |

---

## 5. Training Configuration & Hyperparameters

<a id="table-1"></a>
### Table 1: DAPT Hyperparameter Specifications
| Hyperparameter | Configuration Value | Scientific Rationale |
| :--- | :--- | :--- |
| **Target Steps** | 984 steps | Exactly 3 full passes over 10,484 train sequences |
| **Per-Device Batch Size** | 8 sequences | Memory fitting on 15 GB Tesla T4 GPU |
| **Gradient Accumulation Steps** | 4 steps | Effective batch size of 32 sequences (16,384 tokens) |
| **Peak Learning Rate** | $5.0 \times 10^{-5}$ | Standard continued pretraining rate for ModernBERT |
| **Warmup Schedule** | Linear warmup (59 steps, 6%) | Prevents gradient shock on specialized domain tokens |
| **Decay Schedule** | Monotonic linear decay to 0.0 | Guarantees optimization stability at convergence |
| **Weight Decay** | 0.01 | L2 regularization on transformer weight matrices |
| **MLM Masking Probability** | 15% Bernoulli masking | Standard Devlin et al. BERT objective |
| **Evaluation Frequency** | Every 50 steps | Frequent resolution of validation loss dynamics |
| **Checkpoint Frequency** | Every 100 steps (`save_total_limit=3`) | Disk space management |

---

## 6. Training & Validation Trajectory

<a id="table-4"></a>
### Table 4: Complete Held-Out Validation Trajectory (20 Evaluations)
| Step | MLM Loss | Perplexity | Checkpoint / State Status |
| :---: | :---: | :---: | :--- |
| 50 | 0.9379 | 2.5546 | Periodic Eval |
| 100 | 0.7413 | 2.0987 | Periodic Eval |
| 150 | 0.6837 | 1.9812 | Periodic Eval |
| 200 | 0.6564 | 1.9279 | Periodic Eval |
| 250 | 0.6110 | 1.8423 | Periodic Eval |
| 300 | 0.6106 | 1.8415 | Periodic Eval |
| 350 | 0.6038 | 1.8291 | Periodic Eval |
| 400 | 0.5813 | 1.7883 | Periodic Eval |
| 450 | 0.5718 | 1.7715 | Periodic Eval |
| 500 | 0.5641 | 1.7579 | Periodic Eval |
| 550 | 0.5514 | 1.7357 | Periodic Eval |
| 600 | 0.5555 | 1.7427 | Periodic Eval |
| 650 | 0.5216 | 1.6847 | Periodic Eval |
| 700 | 0.5538 | 1.7399 | Periodic Eval |
| 750 | 0.5251 | 1.6905 | Periodic Eval |
| 800 | 0.5203 | 1.6825 | Periodic Eval |
| 850 | 0.5278 | 1.6951 | Periodic Eval |
| 900 | 0.5158 | 1.6750 | Periodic Eval |
| 950 | 0.4991 | 1.6473 | Best Validation State (unpersisted) |
| 984 | 0.5151 | 1.6739 | Final Training Checkpoint (persisted) |

### Trajectory Analysis:
1. **Rapid Initial Specialization (Steps 50–250)**: MLM loss plunged rapidly from 0.9379 (PPL: 2.5546) to 0.6110 (PPL: 1.8423), capturing frequent maritime syntactic patterns.
2. **Stable Refinement (Steps 250–700)**: Loss decreased steadily from 0.6110 to 0.5216, refining domain technical terminology.
3. **Loss Minimum (Step 950)**: Reached absolute empirical minimum at **MLM Loss = 0.4991** and **Perplexity = 1.6473**.
4. **Terminal Step (Step 984)**: Concluded training at **MLM Loss = 0.5151** and **Perplexity = 1.6739**.

---

## 7. Checkpoint Audit & Artifact Lineage

<a id="table-5"></a>
### Table 5: Model State & Checkpoint Distinction Table
| State / Checkpoint | Step | MLM Loss | PPL | Persisted to Disk? | Status / File Path |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ModernBERT-base Control** | 0 | 1.5365 | 4.6484 | Yes | `baseline-modernbert/evaluation_metrics.json` |
| **Best validation state** | 950 | 0.4991 | 1.6473 | **No** | Minimum observed loss; not saved (`save_steps=100`) |
| **Final training checkpoint** | 984 | 0.5151 | 1.6739 | **Yes** | `dapt/checkpoints/checkpoint-984` & `best/` |
| **Released `MaritimeBERT-v1`** | 984 | 0.5247* | 1.6900* | **Yes** | `dapt/outputs/experiments/MaritimeBERT-v1` |

`*` Evaluated post-training on 585 validation sequences.

### Root-Cause Audit of Checkpoint Management
1. **Why step 950 was not saved**: `eval_steps` was set to 50, but `save_steps` was set to 100. Step 950 was evaluated and flagged as `is_best=True` in memory, but disk serialization was bypassed.
2. **Why step 984 is in `checkpoints/best`**: `dapt/src/training.py` lines 218–225 contained post-loop logic that unconditionally called `save_checkpoint(..., is_best=True)`. This created a copy of `checkpoint-984` inside `dapt/checkpoints/best/`.
3. **Source of `MaritimeBERT-v1`**: Lines 239–244 of `training.py` exported `self.model` (which at the end of training is the **Step 984 model**) into `dapt/outputs/experiments/MaritimeBERT-v1`.
4. **Distinction enforced**: Throughout all reports and figures, Step 950 is designated as **"Best validation state (unpersisted)"**, while Step 984 is designated as **"Final training checkpoint (persisted)"** and the exact source of `MaritimeBERT-v1`.

---

## 8. Post-Training Baseline Comparison

<a id="table-6"></a>
### Table 6: Post-Training Baseline vs. Exported Model Evaluation (585 Packed Val Sequences)
| Evaluation Metric | Baseline ModernBERT | Released MaritimeBERT-v1 | Absolute Delta ($\Delta$) | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Held-Out MLM Loss** | 1.5365 | 0.5247 | -1.0118 | **65.85% loss reduction** |
| **Held-Out Perplexity (PPL)** | 4.6484 | 1.6900 | -2.9584 | **63.64% perplexity reduction** |

---

## 9. Generated Figures

| Figure | PNG Path | Vector PDF Path | Description |
| :--- | :--- | :--- | :--- |
| **Figure 1** | `dapt/figures/validation_loss_curve.png` | `dapt/figures/validation_loss_curve.pdf` | Validation MLM Loss Progression Across DAPT Steps |
| **Figure 2** | `dapt/figures/perplexity_curve.png` | `dapt/figures/perplexity_curve.pdf` | Validation Perplexity Trajectory Across DAPT Steps |
| **Figure 3** | `dapt/figures/baseline_vs_dapt.png` | `dapt/figures/baseline_vs_dapt.pdf` | Post-Training Baseline vs. Exported Model Comparison |
| **Figure 4** | `dapt/figures/learning_rate_trajectory.png` | `dapt/figures/learning_rate_trajectory.pdf` | Learning Rate Warmup and Linear Decay Schedule |
| **Figure 5** | `dapt/figures/checkpoint_comparison.png` | `dapt/figures/checkpoint_comparison.pdf` | Model States & Checkpoint Distinction Architecture |
| **Figure 6** | `dapt/figures/corpus_split_packing_summary.png` | `dapt/figures/corpus_split_packing_summary.pdf` | Corpus Partition and Sequence Packing Summary |

---

## 10. Empirical Artifact Inventory

<a id="table-8"></a>
### Table 8: Artifact Inventory & Locations
| Artifact | Type | Relative Path |
| :--- | :--- | :--- |
| **Raw Training Log** | Text Log | `dapt/log.txt` |
| **Training History JSON** | Parsed Metrics | `dapt/outputs/experiments/training_history.json` |
| **Corpus Manifest** | Data Metadata | `dapt/outputs/data/corpus_manifest.json` |
| **Split Manifest** | Data Splits | `dapt/outputs/data/split_manifest.json` |
| **Tokenizer Diagnostics** | Vocabulary Metadata | `dapt/outputs/data/tokenizer_report.json` |
| **Exported Model Weights** | SafeTensors Model | `dapt/outputs/experiments/MaritimeBERT-v1/model.safetensors` |
| **Exported Model Config** | Model Architecture | `dapt/outputs/experiments/MaritimeBERT-v1/config.json` |
| **Exported Eval Metrics** | Post-Training Eval | `dapt/outputs/experiments/MaritimeBERT-v1/evaluation_metrics.json` |
| **Baseline Eval Metrics** | Baseline Eval | `dapt/outputs/experiments/baseline-modernbert/evaluation_metrics.json` |
| **Comparison Report** | Deltas JSON | `dapt/outputs/experiments/comparison_report.json` |
| **Experiment Summary** | High-Level Results | `dapt/outputs/experiments/dapt_summary.json` |
| **Checkpoint Summary** | Checkpoint Audit | `dapt/outputs/experiments/checkpoint_summary.json` |
| **Execution Notebook** | Executable Notebook | `dapt/runner (1).ipynb` |
