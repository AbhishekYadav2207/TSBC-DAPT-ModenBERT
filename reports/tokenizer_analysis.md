# ModernBERT Tokenizer Diagnostic Report

- **Tokenizer Identifier**: `answerdotai/ModernBERT-base`
- **Vocabulary Size**: 50,368
- **Boundary Token**: `[SEP]` (Token ID `50282`)
- **Sample Document Evaluation**: 10,000 docs (363,690 words -> 592,330 subwords)
- **Empirical Subword Fertility**: **1.6287 subwords/word**
- **Out-of-Vocabulary (`[UNK]`) Rate**: **0.00%** (ModernBERT byte-level BPE ensures 100% token coverage)

## Subword Token Statistics per Document
- **Mean Tokens**: 59.23
- **Median Tokens (P50)**: 54.0
- **P90 Tokens**: 106.0
- **P95 Tokens**: 114.0
- **Max Single Document Tokens**: 145
- **Truncation Rate at 512 Tokens**: **0.00%** (No narrative exceeds 512 subwords)
