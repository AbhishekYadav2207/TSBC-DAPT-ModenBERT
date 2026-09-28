# DAPT Experiment Reproducibility Report: MaritimeBERT-v1

## 1. Provenance & Cryptographic Anchors
- **Input Corpus File**: `maritime_corpus.txt`
- **Corpus SHA-256**: `852fea9a1d756a6989fd55f40646fd7e92ef7597baa4ada36d450e7e8acb6e11`
- **Corpus Size**: 24,556,236 bytes (96,861 documents, 3,830,350 whitespace words, 24,356,820 characters)
- **Deterministic Split Seed**: `42`
- **Split Breakdown**:
  - **Train**: 87,174 docs (3,446,848 words) -> `/content/drive/MyDrive/Abhishek-TSBC/TSBC-Pipeline/dapt/outputs/data/train.txt`
  - **Val**: 4,843 docs (192,475 words) -> `/content/drive/MyDrive/Abhishek-TSBC/TSBC-Pipeline/dapt/outputs/data/val.txt`
  - **Test**: 4,844 docs (191,027 words) -> `/content/drive/MyDrive/Abhishek-TSBC/TSBC-Pipeline/dapt/outputs/data/test.txt`

## 2. Tokenization & Sequence Packing Engine
- **Base Model Architecture**: `answerdotai/ModernBERT-base`
- **Vocab Size**: 50,368
- **Boundary Token**: `[SEP]` (ID `50282`)
- **Tokenizer Fertility**: 1.6287 subwords/word
- **Sequence Block Length**: 512 tokens
- **Packing Strategy**: Concatenated document streams separated by `[SEP]` tokens
- **Train Packing**: 87,174 documents -> 10,484 sequences (Efficiency: 99.99%, Padding waste: 0.01%)
- **Validation Packing**: 4,843 documents -> 585 sequences (Efficiency: 99.89%, Padding waste: 0.11%)

## 3. Hardware & Software Runtime Specifications
- **Operating Platform**: `Linux-6.6.122+-x86_64-with-glibc2.39`
- **Python Version**: `3.13.15 (main, Aug  6 2026, 11:06:22) [GCC 13.3.0]`
- **PyTorch Version**: `2.11.0+cu128`
- **Transformers Version**: `5.16.1`
- **Datasets Version**: `4.8.5`
- **CUDA Runtime**: `12.8` (CUDA Available: `True`)
- **GPU Accelerator**: `Tesla T4` (14.56 GB VRAM, 40 SMs)

## 4. Hyperparameters & Training Loop
- **Per-Device Batch Size**: 8
- **Gradient Accumulation Steps**: 4
- **Effective Batch Size**: 32 sequences (16,384 tokens/step)
- **Target Steps**: 984 steps (3 full epochs)
- **Learning Rate**: 5e-05
- **Warmup Steps**: 59 steps (6% of 984)
- **LR Schedule**: Linear decay to 0.0 at step 984
- **Weight Decay**: 0.01
- **MLM Masking Probability**: 0.15 (Bernoulli distribution)
- **Evaluation Interval**: Every 50 steps (and terminal step 984)
- **Checkpoint Save Interval**: Every 100 steps (save_total_limit = 3)
- **Total Training Elapsed Time**: 6665.35 seconds (~1.85 hours)
- **Average Throughput**: 2415.77 tokens/second

## 5. Checkpoint & Model State Verification
- **Best Validation State**: Step 950 (MLM Loss: 0.4991, Perplexity: 1.6473). **Unpersisted state** (save_steps=100).
- **Final Training Checkpoint**: Step 984 (In-training Loss: 0.5151, Perplexity: 1.6739). **Persisted** at `dapt/checkpoints/checkpoint-984`.
- **Exported Artifact**: `MaritimeBERT-v1` at `dapt/outputs/experiments/MaritimeBERT-v1`, exported from **Step 984 weights**. Post-training evaluation achieves MLM Loss: 0.5247, Perplexity: 1.6900.
