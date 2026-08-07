# 04. Tokenization & Sequence Packing Engine

Tokenization diagnostics and sequence packing operations are executed by [tokenizer.py](file:///d:/CAIR/TSBC-Pipeline/dapt/src/tokenizer.py) and [packing.py](file:///d:/CAIR/TSBC-Pipeline/dapt/src/packing.py).

---

## 1. ModernBERT Tokenizer Diagnostics

### Subword Fertility Rate
Subword tokenization splits words into subword pieces. Fertility measures the average number of subwords generated per word:

$$\text{Fertility} = \frac{\text{Total Subword Tokens}}{\text{Total Source Words}} = \frac{491,332}{301,670} = \mathbf{1.6287 \text{ subwords/word}}$$

- **Base Tokenizer**: `answerdotai/ModernBERT-base`
- **Vocabulary Size**: 50,368 tokens
- **Resolved Boundary Token ID**: `50282` (`sep_token` / `[SEP]`)
- **Unknown Token (`[UNK]`) Rate**: **0.0000%**
- **Truncation Rate ($>512$ tokens)**: **0.00%**

---

## 2. Tokenizer-Aware Document Packing Engine

Because historical maritime narrative documents have a median length of only $51.0$ tokens, feeding documents individually into ModernBERT's max sequence length ($L_{\max}=512$) would waste over $90\%$ of GPU memory on padding tokens.

To solve this, `dapt/src/packing.py` implements continuous subword stream packing:

$$\text{Stream} = \text{Doc}_1 + [\text{SEP}] + \text{Doc}_2 + [\text{SEP}] + \dots + \text{Doc}_N + [\text{SEP}]$$

1. Tokenize each document independently without padding or truncation.
2. Append the resolved boundary token ID (`50282`).
3. Concatenate all subword tokens into a 1D continuous token stream.
4. Chunk the stream into contiguous blocks of length $L_{\max} = 512$.
5. **Efficiency**: Yields **100% active context token utilization**.
