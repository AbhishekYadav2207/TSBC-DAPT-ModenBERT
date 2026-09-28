# 12. Future Research & Extension Roadmap

This document outlines limitations of Version 1 and future research extension vectors.

---

## 1. Limitations of DAPT Version 1

1. **Short Document Length**: Median narrative length of $38.0$ words requires document packing to fill $512$ token blocks.
2. **Template Scaffolding**: Historical maritime logs contain $66.42\%$ boilerplate scaffolding text.
3. **No Custom Vocabulary Surgery**: Version 1 used untouched ModernBERT BPE vocabulary to isolate domain pre-training effects from subword expansion.

---

## 2. Research Roadmap & Experiment Proposals

- **Experiment B (Scaling Pre-Training Duration)**: Extended training from 3 epochs (984 steps) to 10 epochs (~3280 steps) to evaluate asymptotic loss convergence.
- **Experiment C (Scaffolding-Filtered Pre-Training)**: Training exclusively on high-density narrative subsets (filtering out $66.42\%$ boilerplate template text).
- **Experiment D (Custom Vocabulary Expansion)**: Expanding ModernBERT's vocabulary by adding 2,000 domain-specific subword tokens prior to DAPT.
- **Experiment E (Downstream Benchmark Suite)**: Supervised evaluation across Maritime NER, Incident Classification, Extractive QA, and Dense Retrieval tasks.
