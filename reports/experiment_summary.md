# Executive Research Experiment Summary

A publication-ready executive summary of the `MaritimeBERT-v1` Domain-Adaptive Pre-Training experiment.

---

## 1. Executive Summary Table

| Parameter / Metric | Empirical Value | Source Artifact |
| :--- | :--- | :--- |
| **Base Architecture** | `answerdotai/ModernBERT-base` | `dapt/configs/dapt.yaml` |
| **Frozen Input Corpus** | `outputs/maritime_corpus.txt` | `corpus_manifest.json` |
| **SHA-256 Hash Lock** | `b4968819f8b41baa3ee2e2b0e22d103b5d86f5935275a377db51d09ecde3b302` | `corpus_manifest.json` |
| **Total Documents / Words** | 96,715 docs / 3,372,882 words | `corpus_manifest.json` |
| **Train / Val / Test Splits** | 87,043 / 4,835 / 4,837 docs | `split_manifest.json` |
| **Sequence Max Length** | 512 tokens (packed stream) | `tokenizer_report.json` |
| **MLM Masking Ratio** | 15% Bernoulli masking | `dapt/configs/dapt.yaml` |
| **Training Steps / Epochs** | 855 steps / 3.0 epochs | `dapt/configs/dapt.yaml` |
| **Baseline MLM Loss / PPL** | 1.5271 / 4.6050 | `baseline-modernbert/evaluation_metrics.json` |
| **Best Val Checkpoint (Step 800)**| 0.5898 MLM Loss / 1.8037 PPL | `checkpoint-800/step_metrics.json` |
| **Released Model (Step 855)** | 0.5912 MLM Loss / 1.8062 PPL | `comparison_report.json` |
| **Relative Perplexity Gain** | **60.8% PPL Reduction** ($\Delta = -2.7988$) | `comparison_report.json` |
| **Released Export Directory** | `dapt/outputs/experiments/MaritimeBERT-v1` | `dapt/configs/dapt.yaml` |

---

## 2. Main Takeaways

1. **Subsystem Isolation**: DAPT ran as a completely isolated subsystem under strict frozen-corpus immutability rules.
2. **Intrinsic Performance Gain**: DAPT achieved a **60.8% intrinsic reduction in held-out perplexity** ($4.6050 \rightarrow 1.8062$).
3. **Task-Specific Fine-Tuning Required**: The exported backbone `MaritimeBERT-v1` is ready for downstream fine-tuning across NER, classification, extractive QA, and dense retrieval.
