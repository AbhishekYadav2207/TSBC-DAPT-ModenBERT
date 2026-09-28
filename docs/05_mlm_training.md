# 05. 15% Bernoulli MLM Pre-Training Engine

Pre-training execution is managed by [train_dapt.py](file:///d:/CAIR/TSBC-Pipeline/dapt/scripts/train_dapt.py) via [DAPTTrainer](file:///d:/CAIR/TSBC-Pipeline/dapt/src/training.py) and [ModernBERTMaskDataCollator](file:///d:/CAIR/TSBC-Pipeline/dapt/src/masking.py).

---

## 1. 15% Bernoulli MLM Masking

For each packed sequence of length $L=512$, masking positions are selected via a Bernoulli distribution:

$$P(\text{mask}_{i}) = 0.15 \quad \forall i \in \{1, \dots, L\} \setminus \text{SpecialTokens}$$

For selected mask positions $i$:
- $80\%$ of selected positions: Replaced with `[MASK]` token ID.
- $10\%$ of selected positions: Replaced with a random vocabulary token ID.
- $10\%$ of selected positions: Retained unchanged.
- Unmasked target label positions are set to $-100$.

---

## 2. Training Loss & Mathematical Formulation

The Masked Language Modeling loss is computed as Cross-Entropy over masked tokens:

$$\mathcal{L}_{\text{MLM}}(\theta) = -\frac{1}{M} \sum_{m=1}^M \log P(y_m \mid \mathbf{x}_{\text{masked}}; \theta)$$

where $M$ is total masked tokens in the micro-batch.

---

## 3. Micro-Batching & Gradient Accumulation

To achieve an effective batch size of $32$ packed sequences ($16,384$ tokens per update step) without GPU out-of-memory errors:

$$\text{Effective Batch Size} = B_{\text{micro}} \times K_{\text{accum}} = 8 \times 4 = 32 \text{ sequences}$$

$$\mathbf{g}_{\text{acc}} = \frac{1}{4} \sum_{k=1}^4 \nabla_\theta \mathcal{L}_k$$

Total target steps for 3 full epochs over 10,484 sequences:

$$\text{Target Steps} = \left\lceil \frac{10,484 \times 3}{32} \right\rceil = 984 \text{ steps}$$

---

## 4. Optimizer & Scheduler Configuration

- **Optimizer**: `AdamW` ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}, \lambda = 0.01$)
- **Base Learning Rate**: $\eta = 5.0 \times 10^{-5}$
- **Warmup Schedule**: Linear warmup over $6\%$ of total steps (59 steps), followed by linear decay to $0.0$ at terminal step 984.
- **Hardware Acceleration**: Tesla T4 GPU with automatic device placement, processing ~2,416 tokens/sec (~4.7 samples/sec).
