# Dataset Partitioning & Leakage Analysis Report

Derived directly from `dapt/outputs/data/split_manifest.json`.

---

## 1. Partition Distribution

| Split | Document Count | Word Count | Percentage | Saved Location |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | 87,043 | 3,033,940 | $90.0\%$ | `dapt/outputs/data/train.txt` |
| **Validation** | 4,835 | 169,743 | $5.0\%$ | `dapt/outputs/data/val.txt` |
| **Test** | 4,837 | 169,199 | $5.0\%$ | `dapt/outputs/data/test.txt` |

![Split Distribution](file:///d:/CAIR/TSBC-Pipeline/dapt/figures/split_distribution.png)

---

## 2. Data Leakage Diagnostics

- **Exact Duplicate Overlap**: **0 documents** across all splits.
- **Near-Duplicate Overlap (3-Shingle Diagnostic)**: $54.55\%$ validation overlap, $55.75\%$ test overlap due to recurring maritime report preambles.
- **Untouched Test Set**: The 4,837-document test set (`test.txt`) remained completely untouched during all training and validation operations.
