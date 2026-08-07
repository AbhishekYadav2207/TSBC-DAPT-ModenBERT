from typing import List, Dict, Any, Tuple
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from dapt.src.utils import setup_logging, save_json

logger = setup_logging()

def resolve_boundary_token(tokenizer: PreTrainedTokenizerBase) -> Tuple[int, str]:
    """
    Inspects and validates the appropriate document boundary token for the given tokenizer.
    Does NOT assume hardcoded EOS or SEP semantics.
    Fails loudly if no valid boundary token exists.
    """
    if getattr(tokenizer, "eos_token_id", None) is not None:
        return tokenizer.eos_token_id, f"eos_token ({tokenizer.eos_token})"
    if getattr(tokenizer, "sep_token_id", None) is not None:
        return tokenizer.sep_token_id, f"sep_token ({tokenizer.sep_token})"
    if getattr(tokenizer, "cls_token_id", None) is not None:
        return tokenizer.cls_token_id, f"cls_token ({tokenizer.cls_token})"

    raise ValueError(
        f"Unable to determine document boundary token for tokenizer {tokenizer.__class__.__name__}. "
        "No valid eos_token_id, sep_token_id, or cls_token_id found in configuration."
    )

class TokenizerAnalyzer:
    """
    Analyzes pretrained ModernBERT tokenizer performance on the maritime corpus.
    Calculates subword fertility, token length percentiles, and sequence truncation rates.
    """

    def __init__(self, tokenizer_name_or_path: str, trust_remote_code: bool = False):
        logger.info(f"Loading tokenizer from: {tokenizer_name_or_path}")
        self.tokenizer_name = tokenizer_name_or_path
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            trust_remote_code=trust_remote_code
        )
        self.boundary_token_id, self.boundary_token_desc = resolve_boundary_token(self.tokenizer)
        logger.info(f"Resolved document boundary token: {self.boundary_token_desc} (ID: {self.boundary_token_id})")

    def analyze(self, documents: List[str], max_sample_size: int = 10000) -> Dict[str, Any]:
        """
        Computes tokenization diagnostic metrics over documents.
        """
        logger.info(f"Analyzing tokenizer metrics on {len(documents)} documents (sample size cap: {max_sample_size})...")
        sample_docs = documents[:max_sample_size] if len(documents) > max_sample_size else documents
        
        vocab_size = len(self.tokenizer)
        unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
        
        token_lengths = []
        word_counts = []
        total_unk_tokens = 0
        total_subwords = 0

        for doc in sample_docs:
            words = doc.split()
            word_count = len(words)
            word_counts.append(word_count)

            # Tokenize without truncation to measure exact document subword length
            encoded = self.tokenizer.encode(doc, add_special_tokens=True, truncation=False)
            num_tokens = len(encoded)
            token_lengths.append(num_tokens)
            total_subwords += num_tokens

            if unk_token_id is not None:
                total_unk_tokens += encoded.count(unk_token_id)

        arr_tokens = np.array(token_lengths)
        arr_words = np.array(word_counts)
        total_words = int(np.sum(arr_words))

        fertility_subword_per_word = float(total_subwords / total_words) if total_words > 0 else 0.0
        unk_rate = float(total_unk_tokens / total_subwords) if total_subwords > 0 else 0.0

        truncation_128 = sum(1 for l in token_lengths if l > 128)
        truncation_256 = sum(1 for l in token_lengths if l > 256)
        truncation_512 = sum(1 for l in token_lengths if l > 512)
        truncation_1024 = sum(1 for l in token_lengths if l > 1024)

        n_samples = len(sample_docs)
        report = {
            "tokenizer_name_or_path": self.tokenizer_name,
            "vocab_size": vocab_size,
            "boundary_token_id": self.boundary_token_id,
            "boundary_token_description": self.boundary_token_desc,
            "sample_documents": n_samples,
            "total_sample_words": total_words,
            "total_sample_subwords": total_subwords,
            "tokenizer_fertility_subwords_per_word": round(fertility_subword_per_word, 4),
            "unk_rate": round(unk_rate, 6),
            "token_length_statistics": {
                "mean_tokens": round(float(np.mean(arr_tokens)), 2),
                "median_tokens": round(float(np.median(arr_tokens)), 2),
                "p50_tokens": round(float(np.percentile(arr_tokens, 50)), 2),
                "p90_tokens": round(float(np.percentile(arr_tokens, 90)), 2),
                "p95_tokens": round(float(np.percentile(arr_tokens, 95)), 2),
                "max_tokens": int(np.max(arr_tokens)),
            },
            "truncation_counts": {
                "exceeds_128_tokens": truncation_128,
                "exceeds_256_tokens": truncation_256,
                "exceeds_512_tokens": truncation_512,
                "exceeds_1024_tokens": truncation_1024,
            },
            "truncation_rates": {
                "exceeds_128_rate": round(truncation_128 / n_samples, 4),
                "exceeds_256_rate": round(truncation_256 / n_samples, 4),
                "exceeds_512_rate": round(truncation_512 / n_samples, 4),
                "exceeds_1024_rate": round(truncation_1024 / n_samples, 4),
            }
        }

        return report

    def create_report(self, documents: List[str], output_path: str) -> Dict[str, Any]:
        report = self.analyze(documents)
        save_json(report, output_path)
        logger.info(f"Tokenizer report saved to: {output_path}")
        return report
