# 02. Frozen Upstream Corpus Contract

The DAPT subsystem enforces a strict **Frozen Upstream Corpus Contract** to ensure experimental reproducibility, data immutability, and zero contamination.

---

## 1. Frozen Input Specification

The input text corpus resides at the frozen file path:

$$\text{Corpus File Path: } \texttt{outputs/maritime\_corpus.txt}$$

### Cryptographic SHA-256 Hash Lock
Before any dataset preparation or training occurs, `dapt/src/corpus.py` computes the SHA-256 hash of `outputs/maritime_corpus.txt`:

$$\text{SHA-256 Lock: } \texttt{b4968819f8b41baa3ee2e2b0e22d103b5d86f5935275a377db51d09ecde3b302}$$

```text
Upstream Frozen Corpus
          │
  outputs/maritime_corpus.txt
          │
  SHA-256 Verification (b4968819f8b41baa3ee2e2b0e22d103b5d86f5935275a377db51d09ecde3b302)
          │
  Corpus Manifest (corpus_manifest.json)
          │
  Deterministic 90/5/5 Document Split (split_manifest.json)
          │
  Tokenization + Document Packing (tokenizer_report.json)
          │
  ModernBERT 15% Bernoulli MLM Pre-Training
          │
  Validation & Checkpoint Rotation
          │
  Final Held-Out Test & Comparison
          │
  MaritimeBERT-v1 Export
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
| **Total Documents** | **96,715** |
| **Total Words** | **3,372,882** |
| **Total Characters** | **20,671,574** |
| **Unique Vocabulary (Space-Separated)** | **80,174** |
| **Exact Duplicate Lines** | **0** ($0.0\%$) |
| **Mean Word Count** | **34.87** words ($\text{Std} = 15.53$) |
| **Median Word Count** | **37.0** words (P25: 26.0, P75: 45.0, P95: 55.0) |

### Quality Caveats (Preserved in Manifest)
- **Template Scaffolding Ratio**: $66.42\%$ boilerplate scaffolding tokens vs $33.58\%$ domain-derived content tokens.
- **Scaffold-Reduced Near-Duplicate Rate**: $20.58\%$ near-duplicate sentence structures.
- **Pretraining Readiness**: `NEEDS IMPROVEMENT` (noted as an inherent property of historical maritime logs).
