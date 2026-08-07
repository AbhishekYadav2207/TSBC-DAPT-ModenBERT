# 00. DAPT System Overview & Research Motivation

The **Domain-Adaptive Pre-Training (DAPT)** subsystem adapts general-purpose transformer foundation models—specifically `answerdotai/ModernBERT-base`—to specialized technical text in the maritime domain.

---

## 1. Research Motivation

Foundation models trained on generic web corpora (Wikipedia, books, news) struggle with specialized domain terminology, operational syntax, and technical jargon. In maritime occurrence narratives, generic models exhibit high token uncertainty (perplexity) when encountering vessel terminology, navigation acronyms, machinery components, and casualty event logs.

DAPT addresses this domain gap through continued self-supervised pre-training using Masked Language Modeling (MLM) on a domain-specific corpus of 96,715 maritime occurrence narratives (`MaritimeBERT-v1`).

---

## 2. Key Empirical Findings Summary

| Metric | Untouched Baseline (`ModernBERT-base`) | Best Val Checkpoint (Step 800) | Released Model Artifact (Step 855 / `MaritimeBERT-v1`) | Absolute Gain ($\Delta$) | Relative Gain (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Held-Out MLM Loss** | `1.5271` | **`0.5898`** | `0.5912` | $-0.9359$ | **61.3% Loss Reduction** |
| **Held-Out Perplexity** | `4.6050` | **`1.8037`** | `1.8062` | $-2.7988$ | **60.8% PPL Reduction** |

> [!IMPORTANT]
> **Scientific Scope Boundary**: The $60.8\%$ perplexity reduction ($4.6050 \rightarrow 1.8062$) represents an intrinsic held-out language modeling improvement under the self-supervised MLM objective. It demonstrates significantly improved modeling of maritime syntax and vocabulary. It **does not, by itself, establish superiority on downstream tasks** (NER, classification, QA, retrieval, or RAG) without task-specific fine-tuning.

---

## 3. Subsystem Architecture Map

```text
Frozen Upstream Artifact (outputs/maritime_corpus.txt)
          │ SHA-256: b4968819f8b41baa3ee2e2b0e22d103b5d86f5935275a377db51d09ecde3b302
          ▼
1. Cryptographic Ingestion Lock (corpus.py -> corpus_manifest.json)
          │
2. Deterministic 90/5/5 Document Splitting (dataset.py -> split_manifest.json)
          │
3. ModernBERT Tokenizer Diagnostics (tokenizer.py -> tokenizer_report.json)
          │
4. Tokenizer-Aware Document Packing (packing.py -> 512-token uniform blocks)
          │
5. 15% Bernoulli MLM Pre-Training (training.py -> DAPTTrainer)
          │
6. Checkpoint Management & Rotation (checkpointing.py -> Step 800 Best / Step 855 Export)
          │
7. Held-Out Benchmarking (evaluation.py & compare_runs.py -> comparison_report.json)
          │
          ▼
Released Artifact: MaritimeBERT-v1 (dapt/outputs/experiments/MaritimeBERT-v1)
```

---

## 4. Documentation Navigation Index

- [01_architecture.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/01_architecture.md): Subsystem Isolation & Design Principles
- [02_corpus_contract.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/02_corpus_contract.md): Frozen Upstream Corpus Specification & Hash Lock
- [03_data_preparation.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/03_data_preparation.md): Deterministic Splitting & Leakage Diagnostics
- [04_tokenization_packing.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/04_tokenization_packing.md): Subword Fertility & Document Packing Engine
- [05_mlm_training.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/05_mlm_training.md): 15% Bernoulli MLM Masking & DAPTTrainer Loop
- [06_checkpointing.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/06_checkpointing.md): Checkpoint Manager & Step 800 vs Step 855 Distinction
- [07_evaluation.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/07_evaluation.md): Held-Out Evaluation Protocol & Perplexity Formulations
- [08_results.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/08_results.md): Comprehensive Empirical Gain Analysis
- [09_reproducibility.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/09_reproducibility.md): Seed Control, Environment & Reproduction Guide
- [10_downstream_usage.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/10_downstream_usage.md): Task-Specific Fine-Tuning Guide (NER, Classification, QA)
- [11_rag_integration.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/11_rag_integration.md): Retrieval-Augmented Generation Architecture
- [12_future_research.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/12_future_research.md): Limitations & Extended Research Roadmap
- [13_api_reference.md](file:///d:/CAIR/TSBC-Pipeline/dapt/docs/13_api_reference.md): Complete CLI & Source Code API Reference
