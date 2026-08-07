# Empirical Corpus Analysis Report

This report presents empirical findings from the corpus inspection stage executed on `outputs/maritime_corpus.txt` and exported to `dapt/outputs/data/corpus_manifest.json`.

---

## 1. Raw Corpus Metrics

- **Source File Path**: `outputs/maritime_corpus.txt`
- **File Size**: 21,064,178 bytes (~21.06 MB)
- **SHA-256 Hash Lock**: `b4968819f8b41baa3ee2e2b0e22d103b5d86f5935275a377db51d09ecde3b302`
- **Total Document Count**: 96,715 documents
- **Total Word Count**: 3,372,882 words
- **Total Character Count**: 20,671,574 characters
- **Unique Vocabulary (Space-Separated)**: 80,174 tokens
- **Exact Duplicate Count**: 0 duplicate lines ($0.0\%$)

---

## 2. Word Length Distribution

| Word Range Bucket | Document Count | Percentage |
| :--- | :---: | :---: |
| `<20 words` | 18,103 | $18.7\%$ |
| `20–50 words` | 65,366 | $67.6\%$ |
| `50–100 words` | 13,046 | $13.5\%$ |
| `100–200 words` | 176 | $0.18\%$ |
| `200–512 words` | 24 | $0.02\%$ |
| `>512 words` | 0 | $0.0\%$ |

![Corpus Length Distribution](file:///d:/CAIR/TSBC-Pipeline/dapt/figures/corpus_length_distribution.png)

---

## 3. Pretraining Quality Caveats

- **Template Scaffolding Token Ratio**: $66.42\%$ scaffolding vs $33.58\%$ domain content.
- **Scaffold-Reduced Near-Duplicate Rate**: $20.58\%$.
- **Overall Readiness**: `NEEDS IMPROVEMENT` (inherent property of historical maritime logs preserved under the frozen corpus contract).
