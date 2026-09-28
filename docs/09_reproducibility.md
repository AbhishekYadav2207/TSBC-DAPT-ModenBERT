# 09. Reproducibility & Reproduction Protocol

This document provides a step-by-step reproduction protocol to recreate all empirical artifacts from scratch.

---

## 1. Reproducibility Parameters

- **Random Seed**: `42` enforced across PyTorch, NumPy, and Python standard random module.
- **Corpus SHA-256 Hash Lock**: `852fea9a1d756a6989fd55f40646fd7e92ef7597baa4ada36d450e7e8acb6e11`
- **Base Foundation Model**: `answerdotai/ModernBERT-base`
- **PyTorch Version**: `2.11.0+cu128`
- **Transformers Version**: `5.16.1`
- **Datasets Version**: `4.8.5`
- **GPU Accelerator**: Tesla T4 (15.6 GB)

---

## 2. Step-by-Step CLI Reproduction Protocol

```bash
# Step 1: Navigate to repository root
cd /path/to/TSBC-Pipeline

# Step 2: Install DAPT requirements
pip install -r requirements.txt

# Step 3: Ingest Corpus & Generate SHA-256 Lock Manifest
python dapt/scripts/inspect_corpus.py --config dapt/configs/dapt.yaml

# Step 4: Generate Deterministic 90/5/5 Splits
python dapt/scripts/prepare_dataset.py --config dapt/configs/dapt.yaml

# Step 5: ModernBERT Tokenizer Diagnostics
python dapt/scripts/tokenize_corpus.py --config dapt/configs/dapt.yaml

# Step 6: Validate Dataset & Document Sequence Packing
python dapt/scripts/validate_dataset.py --config dapt/configs/dapt.yaml

# Step 7: Evaluate Untouched ModernBERT Baseline
python dapt/scripts/evaluate_dapt.py --config dapt/configs/dapt.yaml --is-baseline --eval-split validation

# Step 8: Execute ModernBERT DAPT Training (984 steps)
python dapt/scripts/train_dapt.py --config dapt/configs/dapt.yaml

# Step 9: Evaluate Trained Model Artifact (MaritimeBERT-v1)
python dapt/scripts/evaluate_dapt.py --config dapt/configs/dapt.yaml --eval-split validation

# Step 10: Compute Metric Deltas
python dapt/scripts/compare_runs.py

# Step 11: Parse Full Training History & Generate Charts & Reports
python dapt/scripts/parse_training_history.py
python dapt/scripts/generate_figures.py
python dapt/scripts/generate_reports.py
python dapt/scripts/validate_documentation.py
pytest dapt/tests/
```
