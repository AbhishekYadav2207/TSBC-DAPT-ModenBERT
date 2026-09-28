# MaritimeBERT Domain-Adaptive Pretraining (DAPT)

An independent, self-contained, research-grade **Domain-Adaptive Pretraining (DAPT)** subsystem for continued Masked Language Modeling (MLM) of `answerdotai/ModernBERT-base` on maritime occurrence narratives (`MaritimeBERT-v1`).

---

## Executive Experiment Summary

| Parameter / Metric | Empirical Value | Source Artifact |
| :--- | :--- | :--- |
| **Base Architecture** | `answerdotai/ModernBERT-base` (149M params) | `dapt/configs/dapt.yaml` |
| **Frozen Input Corpus** | `outputs/maritime_corpus.txt` | `corpus_manifest.json` |
| **SHA-256 Hash Lock** | `852fea9a1d756a6989fd55f40646fd7e92ef7597baa4ada36d450e7e8acb6e11` | `corpus_manifest.json` |
| **Total Documents / Words** | 96,861 docs / 3,830,350 words | `corpus_manifest.json` |
| **Deterministic Splits** | 87,174 Train / 4,843 Val / 4,844 Test (90/5/5) | `split_manifest.json` |
| **Max Sequence Length** | 512 subword tokens (tokenizer-packed stream) | `tokenizer_report.json` |
| **Packed Sequences** | 10,484 Train (99.99% eff) / 585 Val (99.89% eff) | `dapt/log.txt` |
| **MLM Masking Probability** | 15% Bernoulli masking | `dapt/configs/dapt.yaml` |
| **Training Steps / Epochs** | 984 steps / 3.0 full epochs (Effective batch size: 32) | `dapt/log.txt` |
| **Baseline Control Eval** | **1.5365** Loss / **4.6484** Perplexity | `baseline-modernbert/evaluation_metrics.json` |
| **Best Validation State (Step 950)** | **0.4991** Loss / **1.6473** Perplexity | `dapt/log.txt` (Unpersisted state) |
| **Final Checkpoint (Step 984)** | **0.5151** Loss / **1.6739** Perplexity | `checkpoint-984/step_metrics.json` (In-training eval) |
| **Released `MaritimeBERT-v1` (Step 984)** | **0.5247** Loss / **1.6900** Perplexity | `MaritimeBERT-v1/evaluation_metrics.json` (Post-training eval) |
| **Post-Training Relative Gains** | **65.85% Loss Reduction** ($\Delta = -1.0118$) / **63.64% PPL Reduction** ($\Delta = -2.9584$) | `comparison_report.json` |
| **Released Model Location** | `dapt/outputs/experiments/MaritimeBERT-v1` | `dapt/configs/dapt.yaml` |

> [!IMPORTANT]
> **Scientific Scope Boundary**: The 63.64% perplexity reduction ($4.6484 \rightarrow 1.6900$) represents a valid **intrinsic held-out language modeling improvement** under the MLM objective on the 585 packed validation sequences. It demonstrates significantly reduced token uncertainty on specialized maritime terminology. It **does not, by itself, establish downstream task superiority** (NER, Classification, QA, Retrieval, RAG) without task-specific fine-tuning.

---

## Subsystem Architecture & Frozen-Corpus Contract

```text
ModernBERT-base (Untouched Baseline Control)
      │
      │ Frozen Upstream Artifact: outputs/maritime_corpus.txt
      │ SHA-256: 852fea9a1d756a6989fd55f40646fd7e92ef7597baa4ada36d450e7e8acb6e11
      ▼
Deterministic 90/5/5 Document Split ──► Sequence Packing (512 tokens) ──► 15% MLM Training (984 steps)
      │
      ├── Step 950: Best validation state (Loss: 0.4991, PPL: 1.6473) [UNPERSISTED STATE]
      │
      └── Step 984: Final training checkpoint (Loss: 0.5151, PPL: 1.6739) [PERSISTED CHECKPOINT]
            ↓
      MaritimeBERT-v1: Exported Model Weights (from Step 984)
            ↓
      Post-Training Evaluation: MLM Loss = 0.5247, Perplexity = 1.6900
```

- **Read-Only Ingestion**: The source corpus (`outputs/maritime_corpus.txt`) is treated as an immutable research input. DAPT **never** modifies, cleans, rewrites, deduplicates, or performs vocabulary surgery on raw text.
- **Zero Upstream Imports**: The `dapt/` module does not import functions or configs from parent workspace directories.

---

## Documentation Navigation Suite

### 1. Detailed Technical Documentation (`dapt/docs/`)
- [00_overview.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/00_overview.md): System Overview & Research Motivation
- [01_architecture.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/01_architecture.md): Subsystem Architecture & Design Principles
- [02_corpus_contract.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/02_corpus_contract.md): Frozen Upstream Corpus Specification & Hash Lock
- [03_data_preparation.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/03_data_preparation.md): Deterministic Splitting & Leakage Diagnostics
- [04_tokenization_packing.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/04_tokenization_packing.md): ModernBERT Tokenization & Sequence Packing Engine
- [05_mlm_training.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/05_mlm_training.md): 15% Bernoulli MLM Pre-Training Engine
- [06_checkpointing.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/06_checkpointing.md): Checkpointing & Step 950 vs Step 984 Distinction
- [07_evaluation.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/07_evaluation.md): Held-Out Evaluation Protocol & Perplexity Formulations
- [08_results.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/08_results.md): Empirical Gain Analysis & Results
- [09_reproducibility.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/09_reproducibility.md): Reproducibility & Reproduction Protocol
- [10_downstream_usage.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/10_downstream_usage.md): Downstream Task Fine-Tuning Guide (NER, Classification, QA)
- [11_rag_integration.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/11_rag_integration.md): Retrieval-Augmented Generation Architecture
- [12_future_research.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/12_future_research.md): Future Research & Extension Roadmap
- [13_api_reference.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/13_api_reference.md): CLI & Source Code API Reference

### 2. Empirical Analysis Reports (`dapt/reports/`)
- [dapt_report.md](file:///d:/CAIR/TSBC-Pipeline/dapt/reports/dapt_report.md): Comprehensive Primary Technical Report
- [reproducibility_report.md](file:///d:/CAIR/TSBC-Pipeline/dapt/reports/reproducibility_report.md): Complete Reproducibility Specifications
- [corpus_analysis.md](file:///d:/CAIR/TSBC-Pipeline/dapt/reports/corpus_analysis.md): Raw Corpus Metrics & Length Distribution
- [dataset_analysis.md](file:///d:/CAIR/TSBC-Pipeline/dapt/reports/dataset_analysis.md): Split Distribution & Shingle Overlap Leakage
- [tokenizer_analysis.md](file:///d:/CAIR/TSBC-Pipeline/dapt/reports/tokenizer_analysis.md): Fertility Rate & Token Length Statistics
- [training_analysis.md](file:///d:/CAIR/TSBC-Pipeline/dapt/reports/training_analysis.md): Step Trajectory, Loss Curves & Throughput
- [evaluation_analysis.md](file:///d:/CAIR/TSBC-Pipeline/dapt/reports/evaluation_analysis.md): Baseline vs DAPT Metric Deltas
- [experiment_summary.md](file:///d:/CAIR/TSBC-Pipeline/dapt/reports/experiment_summary.md): Publication-Ready Research Summary

### 3. Architecture Diagrams (`dapt/diagrams/`)
- [architecture.md](file:///d:/CAIR/TSBC-Pipeline/dapt/diagrams/architecture.md): Subsystem Components Diagram
- [data_flow.md](file:///d:/CAIR/TSBC-Pipeline/dapt/diagrams/data_flow.md): Ingestion to Packed Token Stream
- [training_pipeline.md](file:///d:/CAIR/TSBC-Pipeline/dapt/diagrams/training_pipeline.md): MLM Forward Pass & Gradient Accumulation
- [checkpoint_lifecycle.md](file:///d:/CAIR/TSBC-Pipeline/dapt/diagrams/checkpoint_lifecycle.md): Checkpoint 950 (Best Val State) vs Step 984 (Final Persisted)
- [downstream_tasks.md](file:///d:/CAIR/TSBC-Pipeline/dapt/diagrams/downstream_tasks.md): Fine-Tuning Adapters
- [rag_architecture.md](file:///d:/CAIR/TSBC-Pipeline/dapt/diagrams/rag_architecture.md): Offline Vector Indexing vs Online LLM Retrieval

---

## Quickstart Reproduction Guide

```bash
# 1. Inspect Corpus & Compute SHA-256 Lock
python dapt/scripts/inspect_corpus.py --config dapt/configs/dapt.yaml

# 2. Prepare Deterministic 90/5/5 Splits
python dapt/scripts/prepare_dataset.py --config dapt/configs/dapt.yaml

# 3. Run ModernBERT Tokenizer Diagnostics
python dapt/scripts/tokenize_corpus.py --config dapt/configs/dapt.yaml

# 4. Validate Dataset & Document Sequence Packing
python dapt/scripts/validate_dataset.py --config dapt/configs/dapt.yaml

# 5. Evaluate Untouched ModernBERT Baseline
python dapt/scripts/evaluate_dapt.py --config dapt/configs/dapt.yaml --is-baseline --eval-split validation

# 6. Execute ModernBERT DAPT Pre-Training (984 steps)
python dapt/scripts/train_dapt.py --config dapt/configs/dapt.yaml

# 7. Evaluate Trained MaritimeBERT-v1 Model
python dapt/scripts/evaluate_dapt.py --config dapt/configs/dapt.yaml --eval-split validation

# 8. Compute Delta Gain Report
python dapt/scripts/compare_runs.py

# 9. Parse Full Training History & Generate Charts
python dapt/scripts/parse_training_history.py
python dapt/scripts/generate_figures.py
python dapt/scripts/generate_reports.py
python dapt/scripts/validate_documentation.py
pytest dapt/tests/
```

---

## Loading Released MaritimeBERT-v1 Model

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_path = "dapt/outputs/experiments/MaritimeBERT-v1"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)
```
