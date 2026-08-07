from typing import List, Dict, Any, Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase
from dapt.src.tokenizer import resolve_boundary_token
from dapt.src.utils import setup_logging

logger = setup_logging()

class DocumentPacker:
    """
    Packs short maritime occurrence documents into uniform fixed-length sequences (max_seq_length)
    using tokenizer-aware boundary token resolution to eliminate padding waste during MLM pretraining.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_seq_length: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.boundary_token_id, self.boundary_desc = resolve_boundary_token(tokenizer)

    def pack_documents(self, documents: List[str]) -> Tuple[List[Dict[str, List[int]]], Dict[str, Any]]:
        """
        Tokenizes documents and concatenates them into max_seq_length chunks with boundary tokens.
        Returns packed dataset samples and efficiency statistics.
        """
        logger.info(f"Packing {len(documents)} documents into max_seq_length={self.max_seq_length} blocks...")
        
        all_token_ids: List[int] = []
        raw_token_count = 0

        for doc in documents:
            # Tokenize document without extra special tokens to manage sequence boundaries explicitly
            encoded = self.tokenizer.encode(doc, add_special_tokens=False, truncation=False)
            if not encoded:
                continue
            
            raw_token_count += len(encoded)
            all_token_ids.extend(encoded)
            all_token_ids.append(self.boundary_token_id)
            raw_token_count += 1

        packed_samples: List[Dict[str, List[int]]] = []
        total_tokens = len(all_token_ids)
        
        for i in range(0, total_tokens, self.max_seq_length):
            chunk = all_token_ids[i:i + self.max_seq_length]
            if len(chunk) < self.max_seq_length:
                # Pad final chunk if necessary
                pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)
                if pad_token_id is None:
                    pad_token_id = 0
                padding_needed = self.max_seq_length - len(chunk)
                attention_mask = [1] * len(chunk) + [0] * padding_needed
                chunk = chunk + [pad_token_id] * padding_needed
            else:
                attention_mask = [1] * self.max_seq_length

            packed_samples.append({
                "input_ids": chunk,
                "attention_mask": attention_mask
            })

        num_sequences = len(packed_samples)
        total_capacity_tokens = num_sequences * self.max_seq_length
        padding_waste_tokens = max(0, total_capacity_tokens - total_tokens)
        packing_efficiency = (total_tokens / total_capacity_tokens) if total_capacity_tokens > 0 else 0.0

        stats = {
            "max_seq_length": self.max_seq_length,
            "boundary_token_id": self.boundary_token_id,
            "boundary_token_description": self.boundary_desc,
            "total_input_documents": len(documents),
            "total_input_tokens": total_tokens,
            "packed_sequence_count": num_sequences,
            "total_capacity_tokens": total_capacity_tokens,
            "padding_waste_tokens": padding_waste_tokens,
            "padding_waste_percent": round((padding_waste_tokens / total_capacity_tokens) * 100, 2) if total_capacity_tokens > 0 else 0.0,
            "packing_efficiency_percent": round(packing_efficiency * 100, 2)
        }

        logger.info(
            f"Packing complete: {len(documents)} docs -> {num_sequences} sequences. "
            f"Efficiency: {stats['packing_efficiency_percent']}% (Padding waste: {stats['padding_waste_percent']}%)"
        )

        return packed_samples, stats
