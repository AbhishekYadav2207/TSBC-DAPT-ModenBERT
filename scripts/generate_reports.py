import os
import json
from pathlib import Path
from typing import Dict, Any

def load_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: dict, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def generate_all_reports() -> Dict[str, str]:
    root = Path(__file__).resolve().parent.parent
    reports_dir = root / "reports"
    experiments_dir = root / "outputs" / "experiments"
    data_dir = root / "outputs" / "data"
    figures_dir = root / "outputs" / "figures"

    reports_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load core artifacts
    history = load_json(experiments_dir / "training_history.json")
    corpus_manifest = load_json(data_dir / "corpus_manifest.json")
    split_manifest = load_json(data_dir / "split_manifest.json")
    tok_report = load_json(data_dir / "tokenizer_report.json")
    comp_report = load_json(experiments_dir / "comparison_report.json")
    eval_metrics = load_json(experiments_dir / "MaritimeBERT-v1" / "evaluation_metrics.json")
    base_metrics = load_json(experiments_dir / "baseline-modernbert" / "evaluation_metrics.json")
    manifest = load_json(experiments_dir / "MaritimeBERT-v1" / "experiment_manifest.json")

    best_state = history["best_validation_state"]
    final_chk = history["final_training_checkpoint"]
    exported_model = history["exported_model"]
    post_comp = history["post_training_comparison"]

    # -------------------------------------------------------------
    # 1. checkpoint_summary.json
    # -------------------------------------------------------------
    chk_summary = {
        "best_validation_state": {
            "step": best_state["step"],
            "epoch": best_state["epoch"],
            "mlm_loss": best_state["mlm_loss"],
            "perplexity": best_state["perplexity"],
            "persisted": False,
            "directory": None,
            "status": "Observed empirical minimum loss state during training; not serialized to disk because save_steps=100"
        },
        "final_training_checkpoint": {
            "step": final_chk["step"],
            "epoch": final_chk["epoch"],
            "in_training_mlm_loss": final_chk["in_training_mlm_loss"],
            "in_training_perplexity": final_chk["in_training_perplexity"],
            "persisted": True,
            "directory": "dapt/checkpoints/checkpoint-984",
            "best_symlink_or_copy": "dapt/checkpoints/best",
            "status": "Final step checkpoint; saved to disk and copied to best/"
        },
        "exported_model": {
            "name": exported_model["name"],
            "source_step": exported_model["source_checkpoint_step"],
            "directory": exported_model["export_directory"],
            "post_training_mlm_loss": exported_model["post_training_mlm_loss"],
            "post_training_perplexity": exported_model["post_training_perplexity"],
            "status": "Exported model weights originating from Step 984; evaluated post-training on held-out validation set"
        },
        "checkpoint_audit_resolution": {
            "inconsistency_found": "The training log recorded 'Updated best checkpoint link' after step 984 despite step 950 having lower validation loss (0.4991 vs 0.5151).",
            "code_root_cause": "In dapt/src/training.py lines 218-225, post-loop logic unconditionally called save_checkpoint(step=global_step, is_best=True), copying step 984 into checkpoints/best. Additionally, save_steps=100 prevented step 950 from being saved to disk.",
            "active_export_origin": "The exported MaritimeBERT-v1 weights correspond strictly to the final Step 984 checkpoint.",
            "correct_terminology": "Step 950 must be referred to as 'Best validation state (unpersisted)', Step 984 as 'Final training checkpoint (persisted)', and MaritimeBERT-v1 as 'Exported model (Step 984)'."
        }
    }
    save_json(chk_summary, experiments_dir / "checkpoint_summary.json")

    # -------------------------------------------------------------
    # 2. dapt_summary.json & dapt_results.json
    # -------------------------------------------------------------
    dapt_sum = {
        "experiment_name": "MaritimeBERT-v1",
        "base_model": "answerdotai/ModernBERT-base",
        "corpus": {
            "file": corpus_manifest["corpus_file"],
            "sha256": corpus_manifest["sha256"],
            "total_documents": corpus_manifest["total_documents"],
            "total_words": corpus_manifest["total_words"],
            "total_characters": corpus_manifest["total_characters"],
            "train_docs": split_manifest["document_counts"]["train"],
            "val_docs": split_manifest["document_counts"]["val"],
            "test_docs": split_manifest["document_counts"]["test"]
        },
        "packing": {
            "max_seq_length": 512,
            "train_sequences": 10484,
            "train_packing_efficiency_pct": 99.99,
            "val_sequences": 585,
            "val_packing_efficiency_pct": 99.89
        },
        "training": {
            "target_steps": 984,
            "per_device_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 32,
            "epochs": 3,
            "learning_rate": 5e-05,
            "warmup_steps": 59,
            "total_duration_seconds": manifest["training_metrics"]["total_training_duration_seconds"],
            "average_throughput_tok_per_sec": manifest["training_metrics"]["average_throughput_tokens_per_sec"]
        },
        "best_validation_state": {
            "step": 950,
            "mlm_loss": 0.4991,
            "perplexity": 1.6473,
            "persisted": False
        },
        "final_training_checkpoint": {
            "step": 984,
            "in_training_mlm_loss": 0.5151,
            "in_training_perplexity": 1.6739,
            "persisted": True
        },
        "exported_model": {
            "name": "MaritimeBERT-v1",
            "source_step": 984,
            "post_training_mlm_loss": 0.5247,
            "post_training_perplexity": 1.6900
        },
        "post_training_baseline_comparison": {
            "baseline_mlm_loss": base_metrics["mlm_loss"],
            "baseline_perplexity": base_metrics["perplexity"],
            "exported_mlm_loss": eval_metrics["mlm_loss"],
            "exported_perplexity": eval_metrics["perplexity"],
            "loss_reduction_pct": 65.85,
            "perplexity_reduction_pct": 63.64,
            "delta_mlm_loss": -1.0118,
            "delta_perplexity": -2.9584
        }
    }
    save_json(dapt_sum, experiments_dir / "dapt_summary.json")
    save_json(dapt_sum, experiments_dir / "dapt_results.json")

    # -------------------------------------------------------------
    # 3. figure_manifest.json & table_manifest.json
    # -------------------------------------------------------------
    fig_manifest = {
        "figures": [
            {
                "id": "figure_1",
                "filename": "validation_loss_curve.png",
                "vector_pdf": "validation_loss_curve.pdf",
                "title": "Validation MLM Loss Progression Across DAPT Steps",
                "caption": "Figure 1: Held-out validation MLM loss progression across all 20 evaluation checkpoints (steps 50–984). The empirical best validation state occurs at Step 950 (loss: 0.4991, unpersisted). The final training checkpoint is Step 984 (in-training loss: 0.5151, persisted and exported)."
            },
            {
                "id": "figure_2",
                "filename": "perplexity_curve.png",
                "vector_pdf": "perplexity_curve.pdf",
                "title": "Validation Perplexity Trajectory Across DAPT Steps",
                "caption": "Figure 2: Held-out validation perplexity trajectory across all 20 evaluated points. Step 950 achieves minimum perplexity (1.6473, unpersisted state); Step 984 final checkpoint reaches perplexity 1.6739."
            },
            {
                "id": "figure_3",
                "filename": "baseline_vs_dapt.png",
                "vector_pdf": "baseline_vs_dapt.pdf",
                "title": "Post-Training Baseline ModernBERT vs. Exported MaritimeBERT-v1 Comparison",
                "caption": "Figure 3: Post-training evaluation comparison on the 585 packed validation sequences. MaritimeBERT-v1 demonstrates a 65.85% reduction in MLM cross-entropy loss (1.5365 -> 0.5247) and a 63.64% reduction in perplexity (4.6484 -> 1.6900)."
            },
            {
                "id": "figure_4",
                "filename": "learning_rate_trajectory.png",
                "vector_pdf": "learning_rate_trajectory.pdf",
                "title": "Training Learning Rate Warmup and Linear Decay Schedule",
                "caption": "Figure 4: Empirical learning rate trajectory showing linear warmup over the first 59 steps to 5e-05, followed by monotonic linear decay terminating at 0.0 at step 984."
            },
            {
                "id": "figure_5",
                "filename": "checkpoint_comparison.png",
                "vector_pdf": "checkpoint_comparison.pdf",
                "title": "Model States & Checkpoint Lifecycle Architecture",
                "caption": "Figure 5: Systematic comparison of all four empirical states: Baseline ModernBERT-base control, Step 950 best validation state (hatched to denote non-persistence), Step 984 final training checkpoint (persisted), and MaritimeBERT-v1 (exported model weights)."
            },
            {
                "id": "figure_6",
                "filename": "corpus_split_packing_summary.png",
                "vector_pdf": "corpus_split_packing_summary.pdf",
                "title": "Maritime Corpus Partition and Sequence Packing Summary",
                "caption": "Figure 6: Corpus partitioning (87,174 train / 4,843 val / 4,844 test) and sequence packing efficiency (99.99% train, 99.89% val) over 512-token blocks."
            }
        ]
    }
    save_json(fig_manifest, figures_dir / "figure_manifest.json")

    tab_manifest = {
        "tables": [
            {
                "id": "table_1",
                "title": "DAPT Configuration and Hyperparameters",
                "location": "dapt/reports/dapt_report.md#table-1"
            },
            {
                "id": "table_2",
                "title": "Maritime Corpus Provenance and Deterministic Splits",
                "location": "dapt/reports/dapt_report.md#table-2"
            },
            {
                "id": "table_3",
                "title": "Sequence Packing and Boundary Efficiency",
                "location": "dapt/reports/dapt_report.md#table-3"
            },
            {
                "id": "table_4",
                "title": "Complete 20-Point Held-Out Validation Trajectory",
                "location": "dapt/reports/dapt_report.md#table-4"
            },
            {
                "id": "table_5",
                "title": "Model State & Checkpoint Distinction Table",
                "location": "dapt/reports/dapt_report.md#table-5"
            },
            {
                "id": "table_6",
                "title": "Post-Training Baseline vs. Exported MaritimeBERT-v1 Comparison",
                "location": "dapt/reports/dapt_report.md#table-6"
            },
            {
                "id": "table_7",
                "title": "System Environment and Reproducibility Specifications",
                "location": "dapt/reports/dapt_report.md#table-7"
            },
            {
                "id": "table_8",
                "title": "Empirical Artifact Inventory and SHA-256 Checksums",
                "location": "dapt/reports/dapt_report.md#table-8"
            }
        ]
    }
    save_json(tab_manifest, figures_dir / "table_manifest.json")

    # -------------------------------------------------------------
    # 4. reproducibility_report.md
    # -------------------------------------------------------------
    repro_md = f"""# DAPT Experiment Reproducibility Report: MaritimeBERT-v1

## 1. Provenance & Cryptographic Anchors
- **Input Corpus File**: `{corpus_manifest['corpus_file']}`
- **Corpus SHA-256**: `{corpus_manifest['sha256']}`
- **Corpus Size**: {corpus_manifest['file_size_bytes']:,} bytes ({corpus_manifest['total_documents']:,} documents, {corpus_manifest['total_words']:,} whitespace words, {corpus_manifest['total_characters']:,} characters)
- **Deterministic Split Seed**: `42`
- **Split Breakdown**:
  - **Train**: {split_manifest['document_counts']['train']:,} docs ({split_manifest['word_counts']['train']:,} words) -> `{split_manifest['file_paths']['train']}`
  - **Val**: {split_manifest['document_counts']['val']:,} docs ({split_manifest['word_counts']['val']:,} words) -> `{split_manifest['file_paths']['val']}`
  - **Test**: {split_manifest['document_counts']['test']:,} docs ({split_manifest['word_counts']['test']:,} words) -> `{split_manifest['file_paths']['test']}`

## 2. Tokenization & Sequence Packing Engine
- **Base Model Architecture**: `{manifest['model']['name_or_path']}`
- **Vocab Size**: {tok_report['vocab_size']:,}
- **Boundary Token**: `[SEP]` (ID `{tok_report['boundary_token_id']}`)
- **Tokenizer Fertility**: {tok_report['tokenizer_fertility_subwords_per_word']:.4f} subwords/word
- **Sequence Block Length**: 512 tokens
- **Packing Strategy**: Concatenated document streams separated by `[SEP]` tokens
- **Train Packing**: {split_manifest['document_counts']['train']:,} documents -> 10,484 sequences (Efficiency: 99.99%, Padding waste: 0.01%)
- **Validation Packing**: {split_manifest['document_counts']['val']:,} documents -> 585 sequences (Efficiency: 99.89%, Padding waste: 0.11%)

## 3. Hardware & Software Runtime Specifications
- **Operating Platform**: `{manifest['system_environment']['platform']}`
- **Python Version**: `{manifest['system_environment']['python_version']}`
- **PyTorch Version**: `{manifest['system_environment']['torch_version']}`
- **Transformers Version**: `{manifest['system_environment']['transformers_version']}`
- **Datasets Version**: `{manifest['system_environment']['datasets_version']}`
- **CUDA Runtime**: `{manifest['system_environment']['cuda_version']}` (CUDA Available: `{manifest['system_environment']['cuda_available']}`)
- **GPU Accelerator**: `{manifest['system_environment']['gpus'][0]['name']}` ({manifest['system_environment']['gpus'][0]['total_memory_gb']:.2f} GB VRAM, {manifest['system_environment']['gpus'][0]['multi_processor_count']} SMs)

## 4. Hyperparameters & Training Loop
- **Per-Device Batch Size**: {manifest['hyperparameters']['batch_size']}
- **Gradient Accumulation Steps**: {manifest['hyperparameters']['gradient_accumulation_steps']}
- **Effective Batch Size**: 32 sequences (16,384 tokens/step)
- **Target Steps**: 984 steps (3 full epochs)
- **Learning Rate**: {manifest['hyperparameters']['learning_rate']}
- **Warmup Steps**: 59 steps (6% of 984)
- **LR Schedule**: Linear decay to 0.0 at step 984
- **Weight Decay**: 0.01
- **MLM Masking Probability**: 0.15 (Bernoulli distribution)
- **Evaluation Interval**: Every 50 steps (and terminal step 984)
- **Checkpoint Save Interval**: Every 100 steps (save_total_limit = 3)
- **Total Training Elapsed Time**: {manifest['training_metrics']['total_training_duration_seconds']:.2f} seconds (~1.85 hours)
- **Average Throughput**: {manifest['training_metrics']['average_throughput_tokens_per_sec']:.2f} tokens/second

## 5. Checkpoint & Model State Verification
- **Best Validation State**: Step 950 (MLM Loss: 0.4991, Perplexity: 1.6473). **Unpersisted state** (save_steps=100).
- **Final Training Checkpoint**: Step 984 (In-training Loss: 0.5151, Perplexity: 1.6739). **Persisted** at `dapt/checkpoints/checkpoint-984`.
- **Exported Artifact**: `MaritimeBERT-v1` at `dapt/outputs/experiments/MaritimeBERT-v1`, exported from **Step 984 weights**. Post-training evaluation achieves MLM Loss: 0.5247, Perplexity: 1.6900.
"""
    with open(reports_dir / "reproducibility_report.md", "w", encoding="utf-8") as f:
        f.write(repro_md)

    # -------------------------------------------------------------
    # 5. dapt_report.md (Comprehensive Primary Publication Report)
    # -------------------------------------------------------------
    val_table_rows = []
    for v in history["validation_history"]:
        step = v["step"]
        loss = v["mlm_loss"]
        ppl = v["perplexity"]
        note = "Best Validation State (unpersisted)" if step == 950 else ("Final Training Checkpoint (persisted)" if step == 984 else "Periodic Eval")
        val_table_rows.append(f"| {step} | {loss:.4f} | {ppl:.4f} | {note} |")
    val_table_md = "\n".join(val_table_rows)

    dapt_report_md = f"""# MaritimeBERT Domain-Adaptive Pretraining (DAPT) - Technical Report

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
> **Scientific Scope Boundary**: The 63.64% perplexity reduction ($4.6484 \\rightarrow 1.6900$) demonstrates valid **intrinsic held-out language modeling improvement** under the MLM self-supervised objective, establishing substantially reduced token uncertainty when predicting maritime technical terminology. It does not by itself assert downstream task superiority (classification, NER, QA, RAG) without fine-tuning adapters.

---

## 2. Environment & System Specifications

<a id="table-7"></a>
### Table 7: System Environment & Hardware Configuration
| Parameter | Value |
| :--- | :--- |
| **Operating System** | {manifest['system_environment']['platform']} |
| **Python Runtime** | {manifest['system_environment']['python_version']} |
| **PyTorch Version** | {manifest['system_environment']['torch_version']} |
| **Transformers Version** | {manifest['system_environment']['transformers_version']} |
| **Datasets Version** | {manifest['system_environment']['datasets_version']} |
| **CUDA Version** | {manifest['system_environment']['cuda_version']} (CUDA Available: {manifest['system_environment']['cuda_available']}) |
| **GPU Accelerator** | {manifest['system_environment']['gpus'][0]['name']} ({manifest['system_environment']['gpus'][0]['total_memory_gb']:.2f} GB VRAM) |
| **Training Duration** | {manifest['training_metrics']['total_training_duration_seconds']:.2f} seconds (6,665.35 s / ~1.85 hours) |
| **Average Throughput** | {manifest['training_metrics']['average_throughput_tokens_per_sec']:.2f} tokens/second (4.72 samples/s) |

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

- **Cryptographic SHA-256 Hash Lock**: `{corpus_manifest['sha256']}`
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
| **Peak Learning Rate** | $5.0 \\times 10^{{-5}}$ | Standard continued pretraining rate for ModernBERT |
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
{val_table_md}

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
"""

    with open(reports_dir / "dapt_report.md", "w", encoding="utf-8") as f:
        f.write(dapt_report_md)

    # Also save to outputs/reports/ for backward compatibility
    outputs_reports_dir = root / "outputs" / "reports"
    outputs_reports_dir.mkdir(parents=True, exist_ok=True)
    with open(outputs_reports_dir / "dapt_report.md", "w", encoding="utf-8") as f:
        f.write(dapt_report_md)

    # -------------------------------------------------------------
    # 6. dapt_report.html (Self-Contained Standalone HTML Report)
    # -------------------------------------------------------------
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MaritimeBERT DAPT Technical Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #24292e; max-width: 960px; margin: 40px auto; padding: 0 20px; }}
h1, h2, h3 {{ color: #1b365d; border-bottom: 1px solid #eaecef; padding-bottom: 8px; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
th, td {{ border: 1px solid #dfe2e5; padding: 8px 12px; text-align: left; }}
th {{ background-color: #f6f8fa; font-weight: 600; }}
tr:nth-child(even) {{ background-color: #fbfcfd; }}
.badge {{ display: inline-block; padding: 2px 8px; font-size: 12px; font-weight: bold; border-radius: 4px; color: #fff; }}
.badge-best {{ background-color: #28a745; }}
.badge-chk {{ background-color: #007bff; }}
.badge-alert {{ background-color: #d73a49; }}
.alert {{ background-color: #fffbdd; border: 1px solid #d1a153; border-radius: 6px; padding: 16px; margin: 20px 0; }}
</style>
</head>
<body>
<h1>MaritimeBERT Domain-Adaptive Pretraining (DAPT) - Technical Report</h1>
<p><strong>Model:</strong> MaritimeBERT-v1 | <strong>Base:</strong> answerdotai/ModernBERT-base (149M params) | <strong>Status:</strong> Validated</p>
<hr>
<h2>1. Executive Summary</h2>
<p>DAPT achieved a <strong>65.85% reduction in held-out MLM loss</strong> (1.5365 &rarr; 0.5247) and a <strong>63.64% reduction in held-out perplexity</strong> (4.6484 &rarr; 1.6900) on 585 validation sequences (299,199 tokens) from TSBC maritime occurrence narratives.</p>

<h3>Core Model States</h3>
<table>
<tr><th>Model / State</th><th>Step</th><th>MLM Loss</th><th>Perplexity</th><th>Persisted?</th><th>Status</th></tr>
<tr><td>ModernBERT-base Baseline</td><td>0</td><td>1.5365</td><td>4.6484</td><td>Yes</td><td>Pretrained control</td></tr>
<tr><td><strong>Best validation state</strong></td><td>950</td><td>0.4991</td><td>1.6473</td><td><span class="badge badge-alert">NO</span></td><td>In-training minimum loss; not saved (save_steps=100)</td></tr>
<tr><td><strong>Final training checkpoint</strong></td><td>984</td><td>0.5151</td><td>1.6739</td><td><span class="badge badge-chk">YES</span></td><td>Persisted in checkpoint-984 & best/</td></tr>
<tr><td><strong>Released MaritimeBERT-v1</strong></td><td>984</td><td>0.5247*</td><td>1.6900*</td><td><span class="badge badge-best">YES</span></td><td>Exported model evaluated post-training</td></tr>
</table>
<p><em>* Post-training evaluation on held-out validation set.</em></p>

<h2>2. Post-Training Baseline Comparison</h2>
<table>
<tr><th>Metric</th><th>ModernBERT Baseline</th><th>MaritimeBERT-v1</th><th>Delta</th><th>Relative Improvement</th></tr>
<tr><td>Held-Out MLM Loss</td><td>1.5365</td><td>0.5247</td><td>-1.0118</td><td><strong>65.85% loss reduction</strong></td></tr>
<tr><td>Held-Out Perplexity</td><td>4.6484</td><td>1.6900</td><td>-2.9584</td><td><strong>63.64% perplexity reduction</strong></td></tr>
</table>

<h2>3. Dataset & Packing Summary</h2>
<ul>
<li><strong>Corpus:</strong> 96,861 documents (3,830,350 words, 24,356,820 characters)</li>
<li><strong>Deterministic Splits:</strong> 87,174 Train (90%) / 4,843 Val (5%) / 4,844 Test (5%)</li>
<li><strong>Sequence Packing:</strong> 10,484 train sequences (99.99% efficiency), 585 val sequences (99.89% efficiency)</li>
<li><strong>Training:</strong> 984 steps, 3 full epochs, effective batch size 32 (16,384 tokens/step)</li>
</ul>
</body>
</html>"""
    with open(reports_dir / "dapt_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    # -------------------------------------------------------------
    # 7. Update existing reports in dapt/reports/
    # -------------------------------------------------------------
    # corpus_analysis.md
    corpus_analysis_md = f"""# Empirical Corpus Analysis Report: Maritime Occurrence Narratives

- **Corpus File**: `{corpus_manifest['corpus_file']}`
- **SHA-256 Checksum**: `{corpus_manifest['sha256']}`
- **File Size**: {corpus_manifest['file_size_bytes']:,} bytes
- **Total Documents**: {corpus_manifest['total_documents']:,}
- **Total Words**: {corpus_manifest['total_words']:,}
- **Total Characters**: {corpus_manifest['total_characters']:,}
- **Unique Vocabulary (Whitespace)**: {corpus_manifest['unique_vocabulary']:,} terms

## Document Length Distribution
- **Mean Words/Document**: {corpus_manifest['length_statistics']['mean_words']:.2f}
- **Median Words/Document (P50)**: {corpus_manifest['length_statistics']['median_words']:.1f}
- **P90 Words**: {corpus_manifest['length_statistics']['p90']:.1f}
- **P95 Words**: {corpus_manifest['length_statistics']['p95']:.1f}
- **Max Words**: {corpus_manifest['length_statistics']['max_words']:,}

## Length Buckets
- `<20 words`: {corpus_manifest['length_buckets']['<20_words']:,} ({corpus_manifest['length_buckets']['<20_words']/corpus_manifest['total_documents']*100:.1f}%)
- `20–50 words`: {corpus_manifest['length_buckets']['20_50_words']:,} ({corpus_manifest['length_buckets']['20_50_words']/corpus_manifest['total_documents']*100:.1f}%)
- `50–100 words`: {corpus_manifest['length_buckets']['50_100_words']:,} ({corpus_manifest['length_buckets']['50_100_words']/corpus_manifest['total_documents']*100:.1f}%)
- `100–200 words`: {corpus_manifest['length_buckets']['100_200_words']:,} ({corpus_manifest['length_buckets']['100_200_words']/corpus_manifest['total_documents']*100:.2f}%)
- `200–512 words`: {corpus_manifest['length_buckets']['200_512_words']:,} ({corpus_manifest['length_buckets']['200_512_words']/corpus_manifest['total_documents']*100:.2f}%)
- `>512 words`: {corpus_manifest['length_buckets']['>512_words']:,}
"""
    with open(reports_dir / "corpus_analysis.md", "w", encoding="utf-8") as f:
        f.write(corpus_analysis_md)

    # dataset_analysis.md
    dataset_analysis_md = f"""# Empirical Dataset Partitioning & Leakage Diagnostic Report

## Split Breakdown (Seed 42)
| Split | Documents | Ratio | Words | File Path |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | {split_manifest['document_counts']['train']:,} | 90.0% | {split_manifest['word_counts']['train']:,} | `dapt/outputs/data/train.txt` |
| **Validation** | {split_manifest['document_counts']['val']:,} | 5.0% | {split_manifest['word_counts']['val']:,} | `dapt/outputs/data/val.txt` |
| **Test** | {split_manifest['document_counts']['test']:,} | 5.0% | {split_manifest['word_counts']['test']:,} | `dapt/outputs/data/test.txt` |
| **Total** | **{split_manifest['document_counts']['total']:,}** | **100.0%** | **{split_manifest['word_counts']['train'] + split_manifest['word_counts']['val'] + split_manifest['word_counts']['test']:,}** | `outputs/maritime_corpus.txt` |

## Leakage Diagnostics
- **Exact Duplicate Overlap**: 0 documents (Train-Val: 0, Train-Test: 0, Val-Test: 0).
- **Near-Duplicate Evaluation**: Evaluated across 2,000 document samples for 8-shingle Jaccard overlap.
"""
    with open(reports_dir / "dataset_analysis.md", "w", encoding="utf-8") as f:
        f.write(dataset_analysis_md)

    # tokenizer_analysis.md
    tokenizer_analysis_md = f"""# ModernBERT Tokenizer Diagnostic Report

- **Tokenizer Identifier**: `{tok_report['tokenizer_name_or_path']}`
- **Vocabulary Size**: {tok_report['vocab_size']:,}
- **Boundary Token**: `[SEP]` (Token ID `{tok_report['boundary_token_id']}`)
- **Sample Document Evaluation**: {tok_report['sample_documents']:,} docs ({tok_report['total_sample_words']:,} words -> {tok_report['total_sample_subwords']:,} subwords)
- **Empirical Subword Fertility**: **{tok_report['tokenizer_fertility_subwords_per_word']:.4f} subwords/word**
- **Out-of-Vocabulary (`[UNK]`) Rate**: **0.00%** (ModernBERT byte-level BPE ensures 100% token coverage)

## Subword Token Statistics per Document
- **Mean Tokens**: {tok_report['token_length_statistics']['mean_tokens']:.2f}
- **Median Tokens (P50)**: {tok_report['token_length_statistics']['median_tokens']:.1f}
- **P90 Tokens**: {tok_report['token_length_statistics']['p90_tokens']:.1f}
- **P95 Tokens**: {tok_report['token_length_statistics']['p95_tokens']:.1f}
- **Max Single Document Tokens**: {tok_report['token_length_statistics']['max_tokens']:,}
- **Truncation Rate at 512 Tokens**: **0.00%** (No narrative exceeds 512 subwords)
"""
    with open(reports_dir / "tokenizer_analysis.md", "w", encoding="utf-8") as f:
        f.write(tokenizer_analysis_md)

    # training_analysis.md
    training_analysis_md = f"""# Empirical Training Trajectory & Performance Analysis

- **Target Steps**: 984 steps (3 full epochs)
- **Effective Batch Size**: 32 sequences (8 per-device x 4 grad accumulation)
- **Elapsed Duration**: {manifest['training_metrics']['total_training_duration_seconds']:.2f} seconds (~1.85 hours)
- **Total Tokens Processed**: {manifest['training_metrics']['total_tokens_processed']:,}
- **Average Throughput**: {manifest['training_metrics']['average_throughput_tokens_per_sec']:.2f} tokens/second

## Trajectory Landmarks
- **Initial Evaluation (Step 50)**: MLM Loss: 0.9379, Perplexity: 2.5546
- **Mid-Point Evaluation (Step 500)**: MLM Loss: 0.5641, Perplexity: 1.7579
- **Best Validation State (Step 950)**: MLM Loss: **0.4991**, Perplexity: **1.6473** (Unpersisted state)
- **Final Training Checkpoint (Step 984)**: MLM Loss: **0.5151**, Perplexity: **1.6739** (Persisted in `checkpoint-984` & `best/`)
- **Exported MaritimeBERT-v1**: Sourced from Step 984. Post-training evaluation achieves MLM Loss: **0.5247**, Perplexity: **1.6900**.
"""
    with open(reports_dir / "training_analysis.md", "w", encoding="utf-8") as f:
        f.write(training_analysis_md)

    # evaluation_analysis.md
    evaluation_analysis_md = f"""# Held-Out Evaluation & Delta Gain Analysis

## Post-Training Baseline vs. Exported Model Evaluation (585 Packed Val Sequences)
| Model | MLM Cross-Entropy Loss | Perplexity (PPL) | Delta vs Baseline |
| :--- | :---: | :---: | :---: |
| **`answerdotai/ModernBERT-base` (Baseline)** | 1.5365 | 4.6484 | — |
| **`MaritimeBERT-v1` (Exported Model)** | 0.5247 | 1.6900 | **-1.0118 Loss (-65.85%) / -2.9584 PPL (-63.64%)** |

## In-Training Validation Trajectory Minimum
- **Best Validation State (Step 950)**: MLM Loss: 0.4991, Perplexity: 1.6473 (Unpersisted state)
- **Final Training Step (Step 984)**: MLM Loss: 0.5151, Perplexity: 1.6739 (Persisted checkpoint)
"""
    with open(reports_dir / "evaluation_analysis.md", "w", encoding="utf-8") as f:
        f.write(evaluation_analysis_md)

    # experiment_summary.md
    experiment_summary_md = f"""# Executive Experiment Summary: MaritimeBERT-v1

- **Base Architecture**: `answerdotai/ModernBERT-base` (149M parameters)
- **Corpus**: `outputs/maritime_corpus.txt` (96,861 documents, 3,830,350 words, SHA-256: `{corpus_manifest['sha256']}`)
- **Splits**: 87,174 Train / 4,843 Val / 4,844 Test (90/5/5)
- **Sequence Packing**: 10,484 Train / 585 Val (Efficiency: 99.99% / 99.89%)
- **Total Steps**: 984 steps (3 full epochs, effective batch size 32)
- **Best Validation State**: Step 950 (MLM Loss: 0.4991, Perplexity: 1.6473, Unpersisted)
- **Final Checkpoint**: Step 984 (In-training Loss: 0.5151, Perplexity: 1.6739, Persisted)
- **Exported Model**: `MaritimeBERT-v1` (Sourced from Step 984; Post-training Loss: 0.5247, Perplexity: 1.6900)
- **Intrinsic Gains (Post-Training Eval)**: **65.85% MLM Loss Reduction** ($1.5365 \\rightarrow 0.5247$) and **63.64% Perplexity Reduction** ($4.6484 \\rightarrow 1.6900$).
"""
    with open(reports_dir / "experiment_summary.md", "w", encoding="utf-8") as f:
        f.write(experiment_summary_md)

    return {
        "checkpoint_summary.json": str(experiments_dir / "checkpoint_summary.json"),
        "dapt_summary.json": str(experiments_dir / "dapt_summary.json"),
        "dapt_results.json": str(experiments_dir / "dapt_results.json"),
        "figure_manifest.json": str(figures_dir / "figure_manifest.json"),
        "table_manifest.json": str(figures_dir / "table_manifest.json"),
        "reproducibility_report.md": str(reports_dir / "reproducibility_report.md"),
        "dapt_report.md": str(reports_dir / "dapt_report.md"),
        "dapt_report.html": str(reports_dir / "dapt_report.html")
    }

if __name__ == "__main__":
    res = generate_all_reports()
    print("Reports generated successfully:")
    for k, v in res.items():
        print(f"  {k} -> {v}")
