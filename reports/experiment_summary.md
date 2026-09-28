# Executive Experiment Summary: MaritimeBERT-v1

- **Base Architecture**: `answerdotai/ModernBERT-base` (149M parameters)
- **Corpus**: `outputs/maritime_corpus.txt` (96,861 documents, 3,830,350 words, SHA-256: `852fea9a1d756a6989fd55f40646fd7e92ef7597baa4ada36d450e7e8acb6e11`)
- **Splits**: 87,174 Train / 4,843 Val / 4,844 Test (90/5/5)
- **Sequence Packing**: 10,484 Train / 585 Val (Efficiency: 99.99% / 99.89%)
- **Total Steps**: 984 steps (3 full epochs, effective batch size 32)
- **Best Validation State**: Step 950 (MLM Loss: 0.4991, Perplexity: 1.6473, Unpersisted)
- **Final Checkpoint**: Step 984 (In-training Loss: 0.5151, Perplexity: 1.6739, Persisted)
- **Exported Model**: `MaritimeBERT-v1` (Sourced from Step 984; Post-training Loss: 0.5247, Perplexity: 1.6900)
- **Intrinsic Gains (Post-Training Eval)**: **65.85% MLM Loss Reduction** ($1.5365 \rightarrow 0.5247$) and **63.64% Perplexity Reduction** ($4.6484 \rightarrow 1.6900$).
