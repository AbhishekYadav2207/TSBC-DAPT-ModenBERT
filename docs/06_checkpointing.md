# 06. Checkpointing & Model Rotation Protocol

Checkpoint management is controlled by [CheckpointManager](file:///d:/CAIR/TSBC-Pipeline/dapt/src/checkpointing.py).

---

## 1. Checkpoint Distinction & Lifecycle

The evaluation framework explicitly distinguishes between intermediate validation states, persisted step checkpoints, and the final released model export:

| Artifact Path / Directory | Step | Held-Out Loss | Held-Out Perplexity | Persisted? | System Role / Meaning |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `answerdotai/ModernBERT-base` | — | `1.5365` | `4.6484` | Yes | Untouched Baseline Control |
| **Best validation state** | **950** | **`0.4991`** | **`1.6473`** | **NO** | **Optimal Validation Loss Minimum (Unpersisted State)** |
| `dapt/checkpoints/checkpoint-800` | 800 | `0.5203` | `1.6825` | Yes | Intermediate Persisted Step Checkpoint |
| `dapt/checkpoints/checkpoint-900` | 900 | `0.5158` | `1.6750` | Yes | Intermediate Persisted Step Checkpoint |
| `dapt/checkpoints/checkpoint-984` | **984** | **`0.5151`** | **`1.6739`** | **Yes** | **Final Training Checkpoint (Persisted)** |
| `dapt/checkpoints/best` | 984 | `0.5151` | `1.6739` | Yes | Directory Copy of Step 984 Checkpoint |
| `dapt/outputs/experiments/MaritimeBERT-v1` | **984** | **`0.5247*`** | **`1.6900*`** | **Yes** | **Released Final Model Export (Weights from Step 984)** |

`*` Post-training evaluation of the exported model on 585 packed validation sequences.

---

## 2. Checkpoint Audit & Root-Cause Explanation

> [!IMPORTANT]
> **Audit Finding on Step 950 vs. Step 984**:
> 1. **Step 950 was not persisted to disk**: Evaluation occurred every 50 steps (`eval_steps = 50`), but checkpoint saving occurred every 100 steps (`save_steps = 100`). At step 950, `val_loss < best_val_loss` was detected in memory (`is_best = True`), but `save_checkpoint()` was not triggered because $950 \pmod{100} \neq 0$. Thus, Step 950 remained an **in-training transient state**.
> 2. **Step 984 is in `checkpoints/best`**: In `dapt/src/training.py` (lines 218–225), post-training-loop logic unconditionally called `self.checkpoint_mgr.save_checkpoint(step=global_step, is_best=True)`. This created a copy of `checkpoint-984` inside `dapt/checkpoints/best/`.
> 3. **`MaritimeBERT-v1` Origin**: Lines 239–244 of `training.py` exported `self.model` (active weights at **Step 984**) into `dapt/outputs/experiments/MaritimeBERT-v1`.
> 4. **Scientific Precision**: Step 950 is accurately documented as the **Best validation state (unpersisted)**, while Step 984 is documented as the **Final training checkpoint (persisted)** and source of `MaritimeBERT-v1`.

---

## 3. Checkpoint Rotation Policy

- **Save Frequency**: Every 100 steps (`save_steps = 100`) and at final step completion.
- **Rotation Limit**: `save_total_limit = 3`, retaining the 3 most recent step checkpoints (`checkpoint-800`, `checkpoint-900`, `checkpoint-984`) plus `checkpoints/best`.
- **Resumption Logic**: Supports resumption via `python dapt/scripts/train_dapt.py --resume-from-checkpoint dapt/checkpoints/checkpoint-900`.
