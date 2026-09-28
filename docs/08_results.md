# 08. Empirical Gain Analysis & Results

Empirical results comparing the untouched ModernBERT baseline against the released `MaritimeBERT-v1` model artifact (derived from `comparison_report.json`).

---

## 1. Empirical Results Table

| Model / Checkpoint | Step | Held-Out Loss | Held-Out Perplexity | Persisted? | Loss Delta ($\Delta$) | Perplexity Delta ($\Delta$) | Status / Protocol |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`ModernBERT-base`** | 0 | `1.5365` | `4.6484` | Yes | baseline | baseline | Baseline Control Eval |
| **Best validation state** | **950** | **`0.4991`** | **`1.6473`** | **NO** | $-1.0374$ | $-3.0011$ | In-training loss minimum (unpersisted) |
| **Final Checkpoint** | **984** | **`0.5151`** | **`1.6739`** | **Yes** | $-1.0214$ | $-2.9745$ | In-training final step eval (persisted) |
| **`MaritimeBERT-v1`** | **984** | **`0.5247*`** | **`1.6900*`** | **Yes** | **$-1.0118$** | **$-2.9584$** | **Post-training exported model eval** |

`*` Evaluated post-training on 585 packed validation sequences.

![Baseline vs DAPT](file:///d:/CAIR/TSBC-Pipeline/dapt/figures/baseline_vs_dapt.png)
![Checkpoint Comparison](file:///d:/CAIR/TSBC-Pipeline/dapt/figures/checkpoint_comparison.png)

---

## 2. Quantitative Deltas: Post-Training Baseline-vs-Exported-Model Comparison

$$\Delta \text{MLM Loss} = 1.5365 - 0.5247 = 1.0118 \quad (\mathbf{65.85\% \text{ Loss Reduction}})$$

$$\Delta \text{Perplexity} = 4.6484 - 1.6900 = 2.9584 \quad (\mathbf{63.64\% \text{ Perplexity Reduction}})$$

---

## 3. Scientific Scope & Boundaries

> [!IMPORTANT]
> **Intrinsic Language Modeling Gain**: The $63.64\%$ reduction in perplexity ($4.6484 \rightarrow 1.6900$) proves superior modeling of maritime occurrence syntax and terminology under the self-supervised MLM objective on the held-out validation set.
>
> **Task Scope Limitation**: This result **does not, by itself, establish superiority on downstream tasks** (NER, classification, QA, retrieval, or RAG) without task-specific fine-tuning.
