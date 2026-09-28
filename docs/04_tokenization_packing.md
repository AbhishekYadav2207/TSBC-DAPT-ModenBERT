# 04. Tokenization & Sequence Packing Engine

Tokenization diagnostics and sequence packing operations are executed by [tokenizer.py](file:///d:/CAIR/TSBC-Pipeline/dapt/src/tokenizer.py) and [packing.py](file:///d:/CAIR/TSBC-Pipeline/dapt/src/packing.py).

---

## 1. ModernBERT Tokenizer Diagnostics

### Subword Fertility Rate
Subword tokenization splits words into subword pieces. Fertility measures the average number of subwords generated per word:

$$\text{Fertility} = \frac{\text{Total Subword Tokens}}{\text{Total Source Words}} = \frac{592,330}{363,690} = \mathbf{1.6287 \text{ subwords/word}}$$

- **Base Tokenizer**: `answerdotai/ModernBERT-base`
- **Vocabulary Size**: 50,368 tokens
- **Resolved Boundary Token ID**: `50282` (`sep_token` / `[SEP]`)
- **Unknown Token (`[UNK]`) Rate**: **0.0000%**
- **Truncation Rate ($>512$ tokens)**: **0.00%**
- **Token Length Distribution**: Mean = 59.23 tokens, Median (P50) = 54.0 tokens, P90 = 106.0 tokens, P95 = 114.0 tokens, Max = 145 tokens

---

## 2. Tokenizer-Aware Document Packing Engine

Because historical maritime narrative documents have a median length of only $54.0$ tokens, feeding documents individually into ModernBERT's max sequence length ($L_{\max}=512$) would waste over $89\%$ of GPU memory on padding tokens.

To solve this, `dapt/src/packing.py` implements continuous subword stream packing:

$$\text{Stream} = \text{Doc}_1 + [\text{SEP}] + \text{Doc}_2 + [\text{SEP}] + \dots + \text{Doc}_N + [\text{SEP}]$$

1. Tokenize each document independently without padding or truncation.
2. Append the resolved boundary token ID (`50282`).
3. Concatenate all subword tokens into a 1D continuous token stream.
4. Chunk the stream into contiguous blocks of length $L_{\max} = 512$.
5. **Efficiency Achieved**:
   - **Train Set**: 87,174 documents packed into **10,484 sequences** of length 512 (Efficiency: **99.99%**, Padding waste: **0.01%**).
   - **Validation Set**: 4,843 documents packed into **585 sequences** of length 512 (Efficiency: **99.89%**, Padding waste: **0.11%**).
