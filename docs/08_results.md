# 08. Empirical Gain Analysis & Results

Empirical results comparing the untouched ModernBERT baseline against the released `MaritimeBERT-v1` model artifact (derived from `comparison_report.json`).

---

## 1. Empirical Results Table

| Model / Checkpoint | Step | Held-Out Loss | Held-Out Perplexity | Loss Delta ($\Delta$) | Perplexity Delta ($\Delta$) | Relative Gain (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ModernBERT-base`** | — | `1.5271` | `4.6050` | baseline | baseline | baseline |
| **Best Val Checkpoint** | **800** | **`0.5898`** | **`1.8037`** | $-0.9373$ | $-2.8013$ | **60.8% PPL Gain** |
| **`MaritimeBERT-v1`** | **855** | **`0.5912`** | **`1.8062`** | $-0.9359$ | $-2.7988$ | **60.8% PPL Gain** |

![Baseline vs DAPT](file:///d:/CAIR/TSBC-Pipeline/dapt/figures/baseline_vs_dapt.png)
![Checkpoint Comparison](file:///d:/CAIR/TSBC-Pipeline/dapt/figures/checkpoint_comparison.png)

---

## 2. Quantitative Deltas

$$\Delta \text{MLM Loss} = 1.5271 - 0.5912 = 0.9359 \quad (\mathbf{61.3\% \text{ Loss Reduction}})$$

$$\Delta \text{Perplexity} = 4.6050 - 1.8062 = 2.7988 \quad (\mathbf{60.8\% \text{ Perplexity Reduction}})$$

---

## 3. Scientific Scope & Boundaries

> [!IMPORTANT]
> **Intrinsic Language Modeling Gain**: The $60.8\%$ reduction in perplexity ($4.6050 \rightarrow 1.8062$) proves superior modeling of maritime occurrence syntax and terminology under the self-supervised MLM objective.
>
> **Task Scope Limitation**: This result **does not, by itself, establish superiority on downstream tasks** (NER, classification, QA, retrieval, or RAG) without task-specific fine-tuning.
