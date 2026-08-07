# 09. Reproducibility & Reproduction Protocol

This document provides a step-by-step reproduction protocol to recreate all empirical artifacts from scratch.

---

## 1. Reproducibility Parameters

- **Random Seed**: `42` enforced across PyTorch, NumPy, and Python standard random module.
- **Corpus SHA-256 Hash Lock**: `b4968819f8b41baa3ee2e2b0e22d103b5d86f5935275a377db51d09ecde3b302`
- **Base Foundation Model**: `answerdotai/ModernBERT-base`
- **PyTorch Version**: `torch >= 2.0.0`
- **Transformers Version**: `transformers >= 4.40.0`

---

## 2. Step-by-Step CLI Reproduction Protocol

```bash
# Step 1: Navigate to repository root
cd /path/to/TSBC-Pipeline

# Step 2: Install DAPT requirements
pip install -r dapt/requirements.txt

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

# Step 8: Execute ModernBERT DAPT Training (855 steps)
python dapt/scripts/train_dapt.py --config dapt/configs/dapt.yaml

# Step 9: Evaluate Trained Model Artifact
python dapt/scripts/evaluate_dapt.py --config dapt/configs/dapt.yaml --eval-split validation

# Step 10: Compute Metric Deltas
python dapt/scripts/compare_runs.py

# Step 11: Generate Charts & Run Documentation Validation
python dapt/scripts/generate_figures.py
python dapt/scripts/validate_documentation.py
pytest dapt/tests/
```
