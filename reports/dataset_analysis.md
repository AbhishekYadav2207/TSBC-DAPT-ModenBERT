# Empirical Dataset Partitioning & Leakage Diagnostic Report

## Split Breakdown (Seed 42)
| Split | Documents | Ratio | Words | File Path |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | 87,174 | 90.0% | 3,446,848 | `dapt/outputs/data/train.txt` |
| **Validation** | 4,843 | 5.0% | 192,475 | `dapt/outputs/data/val.txt` |
| **Test** | 4,844 | 5.0% | 191,027 | `dapt/outputs/data/test.txt` |
| **Total** | **96,861** | **100.0%** | **3,830,350** | `outputs/maritime_corpus.txt` |

## Leakage Diagnostics
- **Exact Duplicate Overlap**: 0 documents (Train-Val: 0, Train-Test: 0, Val-Test: 0).
- **Near-Duplicate Evaluation**: Evaluated across 2,000 document samples for 8-shingle Jaccard overlap.
