# Held-Out Evaluation & Delta Gain Analysis

## Post-Training Baseline vs. Exported Model Evaluation (585 Packed Val Sequences)
| Model | MLM Cross-Entropy Loss | Perplexity (PPL) | Delta vs Baseline |
| :--- | :---: | :---: | :---: |
| **`answerdotai/ModernBERT-base` (Baseline)** | 1.5365 | 4.6484 | — |
| **`MaritimeBERT-v1` (Exported Model)** | 0.5247 | 1.6900 | **-1.0118 Loss (-65.85%) / -2.9584 PPL (-63.64%)** |

## In-Training Validation Trajectory Minimum
- **Best Validation State (Step 950)**: MLM Loss: 0.4991, Perplexity: 1.6473 (Unpersisted state)
- **Final Training Step (Step 984)**: MLM Loss: 0.5151, Perplexity: 1.6739 (Persisted checkpoint)
