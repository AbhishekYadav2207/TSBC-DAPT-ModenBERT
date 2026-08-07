# Tokenizer Diagnostics Analysis Report

Derived directly from `dapt/outputs/data/tokenizer_report.json`.

---

## 1. ModernBERT Tokenizer Statistics

- **Base Tokenizer**: `answerdotai/ModernBERT-base`
- **Vocabulary Size**: 50,368 tokens
- **Resolved Boundary Token ID**: `50282` (`sep_token` / `[SEP]`)
- **Subword Fertility Rate**: **1.6287 subwords per word**
- **Unknown Token (`[UNK]`) Rate**: **0.0000%**

---

## 2. Token Length Distribution

- **Mean Sequence Tokens**: $49.13$
- **Median (P50) Tokens**: $51.0$
- **P90 Tokens**: $75.0$
- **P95 Tokens**: $81.0$
- **Max Single Document Tokens**: $102$
- **Truncation Rate ($>512$ tokens)**: **0.00%**

![Tokenizer Fertility & Length](file:///d:/CAIR/TSBC-Pipeline/dapt/figures/tokenizer_fertility.png)
