# 03. Deterministic Dataset Splitting & Leakage Diagnostics

Dataset preparation is executed by [prepare_dataset.py](file:///d:/CAIR/TSBC-Pipeline/dapt/scripts/prepare_dataset.py) using [dataset.py](file:///d:/CAIR/TSBC-Pipeline/dapt/src/dataset.py).

---

## 1. Document-Level Splitting Protocol

To prevent evaluation leakage, dataset partitioning is performed strictly at the **document (narrative line) level** rather than sequence level.

- **Split Ratios**: Train: $90\%$, Validation: $5\%$, Test: $5\%$
- **Split Seed**: `42`

### Dataset Partition Quantities (Derived from `split_manifest.json`)

| Partition | Document Count | Word Count | Percentage | Saved Path |
| :--- | :---: | :---: | :---: | :--- |
| **Train Set** | **87,043** | 3,033,940 | $90.0\%$ | `dapt/outputs/data/train.txt` |
| **Validation Set** | **4,835** | 169,743 | $5.0\%$ | `dapt/outputs/data/val.txt` |
| **Test Set** | **4,837** | 169,199 | $5.0\%$ | `dapt/outputs/data/test.txt` |
| **Total** | **96,715** | **3,372,882** | **$100.0\%$** | — |

---

## 2. Data Leakage Diagnostics

`dapt/src/dataset.py` executes exact line matching and 3-shingle overlap diagnostics across all splits.

```json
{
  "exact_duplicate_leakage": {
    "train_val_overlap": 0,
    "train_test_overlap": 0,
    "val_test_overlap": 0
  },
  "near_duplicate_leakage_diagnostic": {
    "sample_evaluated": 2000,
    "val_high_shingle_overlap_count": 1091,
    "val_high_shingle_overlap_rate": 0.5455,
    "test_high_shingle_overlap_count": 1115,
    "test_high_shingle_overlap_rate": 0.5575
  }
}
```

### Analysis of Leakage Results
- **Exact Duplicate Leakage**: **0 document overlap** between train, validation, and test sets.
- **Near-Duplicate Overlap**: High 3-shingle overlap ($54.55\%$ validation, $55.75\%$ test) caused by recurring maritime safety report templates (e.g., standard vessel casualty preamble boilerplate).

---

## 3. Test Set Preservation

> [!IMPORTANT]
> The **4,837-document test set (`test.txt`) remains completely untouched** during training and hyperparameter tuning. It is reserved exclusively for final un-biased benchmarking.
