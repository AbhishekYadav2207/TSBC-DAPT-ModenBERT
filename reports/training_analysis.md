# Empirical Training Trajectory & Performance Analysis

- **Target Steps**: 984 steps (3 full epochs)
- **Effective Batch Size**: 32 sequences (8 per-device x 4 grad accumulation)
- **Elapsed Duration**: 6665.35 seconds (~1.85 hours)
- **Total Tokens Processed**: 16,101,987
- **Average Throughput**: 2415.77 tokens/second

## Trajectory Landmarks
- **Initial Evaluation (Step 50)**: MLM Loss: 0.9379, Perplexity: 2.5546
- **Mid-Point Evaluation (Step 500)**: MLM Loss: 0.5641, Perplexity: 1.7579
- **Best Validation State (Step 950)**: MLM Loss: **0.4991**, Perplexity: **1.6473** (Unpersisted state)
- **Final Training Checkpoint (Step 984)**: MLM Loss: **0.5151**, Perplexity: **1.6739** (Persisted in `checkpoint-984` & `best/`)
- **Exported MaritimeBERT-v1**: Sourced from Step 984. Post-training evaluation achieves MLM Loss: **0.5247**, Perplexity: **1.6900**.
