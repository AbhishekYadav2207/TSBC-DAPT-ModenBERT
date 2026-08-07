# DAPT Training Analysis Report

Derived from training execution logs, `dapt/configs/dapt.yaml`, and checkpoint step metrics.

---

## 1. Hyperparameter Configuration

- **Target Epochs**: $3.0$
- **Total Training Steps**: $855$ steps
- **Per-Device Batch Size**: $8$
- **Gradient Accumulation Steps**: $4$
- **Effective Batch Size**: $32$ packed sequences ($16,384$ tokens per update step)
- **Base Learning Rate**: $\eta = 5.0 \times 10^{-5}$
- **Optimizer**: `AdamW` ($\beta_1=0.9, \beta_2=0.999, \lambda=0.01$)
- **Scheduler**: Linear decay with $6\%$ warmup (~51 steps)

---

## 2. Loss & Perplexity Trajectory

| Step | Epoch | Held-Out MLM Loss | Held-Out Perplexity | System Role / Event |
| :---: | :---: | :---: | :---: | :--- |
| **0** | 0 | `1.5271` | `4.6050` | Untouched Baseline Control |
| **700** | 2.45 | `0.5916` | `1.8069` | Intermediate Checkpoint |
| **800** | 2.80 | **`0.5898`** | **`1.8037`** | **Optimal Validation Loss Checkpoint** |
| **855** | 3.00 | **`0.5912`** | **`1.8062`** | **Released Exported Artifact (`MaritimeBERT-v1`)** |

![Validation Loss Curve](file:///d:/CAIR/TSBC-Pipeline/dapt/figures/validation_loss_curve.png)
![Perplexity Curve](file:///d:/CAIR/TSBC-Pipeline/dapt/figures/perplexity_curve.png)
