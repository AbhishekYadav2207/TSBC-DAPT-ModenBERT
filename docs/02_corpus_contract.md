# 02. Frozen Upstream Corpus Contract

The DAPT subsystem enforces a strict **Frozen Upstream Corpus Contract** to ensure experimental reproducibility, data immutability, and zero contamination.

---

## 1. Frozen Input Specification

The input text corpus resides at the frozen file path:

$$\text{Corpus File Path: } \texttt{outputs/maritime\_corpus.txt}$$

### Cryptographic SHA-256 Hash Lock
Before any dataset preparation or training occurs, `dapt/src/corpus.py` computes the SHA-256 hash of `outputs/maritime_corpus.txt`:

$$\text{SHA-256 Lock: } \texttt{852fea9a1d756a6989fd55f40646fd7e92ef7597baa4ada36d450e7e8acb6e11}$$

```text
Upstream Frozen Corpus
          │
  outputs/maritime_corpus.txt
          │
  SHA-256 Verification (852fea9a1d756a6989fd55f40646fd7e92ef7597baa4ada36d450e7e8acb6e11)
          │
  Corpus Manifest (corpus_manifest.json)
          │
  Deterministic 90/5/5 Document Split (split_manifest.json)
          │
  Tokenization + Document Packing (tokenizer_report.json)
          │
  ModernBERT 15% Bernoulli MLM Pre-Training (984 steps)
          │
  Validation & Checkpoint Rotation (Step 950 best state / Step 984 final checkpoint)
          │
  MaritimeBERT-v1 Export & Held-Out Comparison
```

---

## 2. Immutable Corpus Rules

1. **No Text Rewriting**: Raw lines in `outputs/maritime_corpus.txt` are ingested exactly as written.
2. **No Post-hoc Deduplication**: DAPT does not remove or filter lines during pre-training.
3. **No Synthetic Augmentation**: No paraphrasing or masking modification is applied to raw source text.
4. **No Vocabulary Surgery**: DAPT Version 1 uses the untouched ModernBERT vocabulary to strictly isolate domain pre-training effects from subword vocabulary expansion.

---

## 3. Empirically Measured Corpus Statistics

Derived directly from `dapt/outputs/data/corpus_manifest.json`:

| Property | Value |
| :--- | :--- |
| **Total Documents** | **96,861** |
| **Total Words** | **3,830,350** |
| **Total Characters** | **24,356,820** |
| **Unique Vocabulary (Space-Separated)** | **80,333** |
| **Exact Duplicate Documents** | **0** ($0.0\%$) |
| **Mean Word Count** | **39.54** words ($\text{Std} = 20.80$) |
| **Median Word Count** | **38.0** words (P25: 26.0, P75: 52.0, P95: 75.0) |
| **Max Word Count** | **514** words |
