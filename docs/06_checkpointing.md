# 06. Checkpointing & Model Rotation Protocol

Checkpoint management is controlled by [CheckpointManager](file:///d:/CAIR/TSBC-Pipeline/dapt/src/checkpointing.py).

---

## 1. Checkpoint Distinction & Lifecycle

The evaluation framework explicitly distinguishes between intermediate validation checkpoints and the final released model export:

| Artifact Path / Directory | Step | Held-Out Loss | Held-Out Perplexity | System Role / Meaning |
| :--- | :---: | :---: | :---: | :--- |
| `answerdotai/ModernBERT-base` | — | `1.5271` | `4.6050` | Untouched Baseline Control |
| `dapt/checkpoints/checkpoint-700` | 700 | `0.5916` | `1.8069` | Intermediate Step Checkpoint |
| `dapt/checkpoints/checkpoint-800` | **800** | **`0.5898`** | **`1.8037`** | **Optimal Validation Checkpoint (Best Loss)** |
| `dapt/checkpoints/checkpoint-855` | 855 | `0.5996` | `1.8213` | Step 855 Final Training Step Checkpoint |
| `dapt/checkpoints/best` | 800 | `0.5898` | `1.8037` | Directory Link to Best Validation Checkpoint |
| `dapt/outputs/experiments/MaritimeBERT-v1` | **855** | **`0.5912`** | **`1.8062`** | **Released Final Model Export** |

> [!NOTE]
> **Checkpoint Distinction**: Step 800 achieved the lowest validation loss (`0.5898`). Step 855 represents the completed 3-epoch model weights exported as the `MaritimeBERT-v1` release artifact.

---

## 2. Checkpoint Rotation Policy

- **Save Frequency**: Every 100 steps (`save_steps = 100`) and at final step completion.
- **Rotation Limit**: `save_total_limit = 3`, retaining only the 3 most recent step checkpoints plus the best validation checkpoint (`checkpoints/best`).
- **Resumption Logic**: Supports resuming via `python dapt/scripts/train_dapt.py --resume-from-checkpoint dapt/checkpoints/checkpoint-800`.
