# Evaluation & Baseline Gain Analysis Report

Derived directly from `dapt/outputs/experiments/comparison_report.json` and checkpoint evaluation metrics.

---

## 1. Empirical Gain Summary

| Model / Checkpoint | Step | MLM Loss | Perplexity | System Role |
| :--- | :---: | :---: | :---: | :--- |
| **`answerdotai/ModernBERT-base`** | — | `1.5271` | `4.6050` | Untouched Baseline Control |
| **DAPT Best Validation Checkpoint** | **Step 800** | **`0.5898`** | **`1.8037`** | Min Validation Loss Checkpoint |
| **MaritimeBERT-v1 Released Model** | **Step 855** | **`0.5912`** | **`1.8062`** | Final Released Model Export |

---

## 2. Intrinsic Performance Gains ($\Delta$)

$$\Delta \text{MLM Loss} = 1.5271 - 0.5912 = 0.9359 \quad (\mathbf{61.3\% \text{ Loss Reduction}})$$

$$\Delta \text{Perplexity} = 4.6050 - 1.8062 = 2.7988 \quad (\mathbf{60.8\% \text{ Perplexity Reduction}})$$

![Baseline vs DAPT](file:///d:/CAIR/TSBC-Pipeline/dapt/figures/baseline_vs_dapt.png)
![Checkpoint Comparison](file:///d:/CAIR/TSBC-Pipeline/dapt/figures/checkpoint_comparison.png)

---

## 3. Scientific Scope Boundary Reminder

> These are intrinsic held-out MLM improvements. They demonstrate significantly improved language modeling of maritime text under the self-supervised MLM objective. They **do not establish downstream NER, classification, QA, retrieval, or RAG superiority without task-specific fine-tuning.**
