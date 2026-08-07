from typing import List, Dict, Any, Union
import torch
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase

class ModernBERTMaskDataCollator:
    """
    Data Collator for Masked Language Modeling (MLM) pretraining.
    Wraps DataCollatorForLanguageModeling or implements random masking for PyTorch tensors.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, mlm_probability: float = 0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability
        )

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collates a batch of examples and applies MLM masking.
        """
        # Convert dictionary inputs to tensor format if needed
        batch = self.collator(examples)
        return batch
