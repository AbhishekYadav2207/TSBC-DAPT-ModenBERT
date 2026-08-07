# 07. Held-Out Evaluation Protocol & Perplexity Formulations

Evaluation is managed by [MLMEvaluator](file:///d:/CAIR/TSBC-Pipeline/dapt/src/evaluation.py) and [evaluate_dapt.py](file:///d:/CAIR/TSBC-Pipeline/dapt/scripts/evaluate_dapt.py).

---

## 1. Held-Out Evaluation Protocol

- **Validation Split**: `val.txt` (4,835 documents, 169,743 words, 511 packed 512-token evaluation samples).
- **Test Split**: `test.txt` (4,837 documents, 169,199 words - held out untouched).
- **Evaluation Batch Size**: $8$ samples per device.
- **Masking Probability**: $15\%$ Bernoulli masking.

---

## 2. Mathematical Formulations

### Masked Cross-Entropy Loss
$$\mathcal{L}_{\text{MLM}} = -\frac{1}{M} \sum_{m=1}^M \log P(y_m \mid \mathbf{x}_{\text{masked}})$$

where $M$ is total evaluated masked tokens (~38,300 tokens per evaluation run).

### Perplexity (PPL)
$$\text{PPL} = \exp(\mathcal{L}_{\text{MLM}})$$

Perplexity measures the model's exponentiated average token uncertainty. A reduction from $4.6050$ to $1.8062$ indicates a reduction in average subword token search space uncertainty from ~4.6 candidate subwords down to ~1.8 subwords per masked position.
