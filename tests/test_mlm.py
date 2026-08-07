import torch
from transformers import AutoTokenizer
from dapt.src.masking import ModernBERTMaskDataCollator

def test_mlm_data_collator():
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    collator = ModernBERTMaskDataCollator(tokenizer, mlm_probability=0.15)

    samples = [
        {"input_ids": [101, 2000, 2001, 102] + [0] * 124, "attention_mask": [1, 1, 1, 1] + [0] * 124},
        {"input_ids": [101, 3000, 3001, 102] + [0] * 124, "attention_mask": [1, 1, 1, 1] + [0] * 124},
    ]

    batch = collator(samples)
    assert "input_ids" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape == (2, 128)
    assert batch["labels"].shape == (2, 128)
