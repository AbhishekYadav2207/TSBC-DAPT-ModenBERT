import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from dapt.src.utils import resolve_path, save_json, setup_logging

logger = setup_logging()

class CorpusLoader:
    """
    Immutable loader for the maritime training corpus.
    Calculates cryptographic SHA-256 hashes, length statistics, duplicate counts,
    and produces a machine-readable corpus manifest.
    """

    def __init__(self, corpus_path: str):
        self.corpus_file = resolve_path(corpus_path)
        if not self.corpus_file.exists():
            raise FileNotFoundError(f"Corpus file does not exist: {self.corpus_file}")
        if not os.access(self.corpus_file, os.R_OK):
            raise PermissionError(f"Corpus file is not readable: {self.corpus_file}")

    def compute_sha256(self, chunk_size: int = 65536) -> str:
        """
        Computes SHA-256 hash of the input corpus file without loading entire file into RAM at once.
        """
        hasher = hashlib.sha256()
        with open(self.corpus_file, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()

    def load_documents(self) -> List[str]:
        """
        Reads documents from the corpus file. Documents consist of one or more non-empty lines
        and are separated by blank lines. Non-empty lines within each document are joined using a single space.
        """
        documents = []
        with open(self.corpus_file, "r", encoding="utf-8", errors="replace") as f:
            current_doc_lines = []
            for line in f:
                stripped = line.strip()
                if stripped:
                    current_doc_lines.append(stripped)
                else:
                    if current_doc_lines:
                        documents.append(" ".join(current_doc_lines))
                        current_doc_lines = []
            if current_doc_lines:
                documents.append(" ".join(current_doc_lines))
        return documents

    def analyze_corpus(self) -> Dict[str, Any]:
        """
        Computes canonical corpus metadata and statistics.
        """
        logger.info(f"Analyzing corpus at: {self.corpus_file}")
        sha256_hash = self.compute_sha256()
        documents = self.load_documents()
        
        file_size_bytes = self.corpus_file.stat().st_size
        total_docs = len(documents)
        total_chars = sum(len(doc) for doc in documents)
        
        doc_word_counts = [len(doc.split()) for doc in documents]
        total_words = sum(doc_word_counts)
        
        # Unique vocabulary estimate (whitespace tokenized)
        vocab_set = set()
        for doc in documents:
            vocab_set.update(doc.lower().split())
        unique_vocab = len(vocab_set)

        # Duplicate analysis
        doc_set = set()
        exact_duplicates = 0
        for doc in documents:
            if doc in doc_set:
                exact_duplicates += 1
            else:
                doc_set.add(doc)
        exact_duplicate_rate = (exact_duplicates / total_docs) if total_docs > 0 else 0.0

        # Percentiles & Length Stats
        if doc_word_counts:
            arr = np.array(doc_word_counts)
            mean_len = float(np.mean(arr))
            median_len = float(np.median(arr))
            std_len = float(np.std(arr))
            min_len = int(np.min(arr))
            max_len = int(np.max(arr))
            p10 = float(np.percentile(arr, 10))
            p25 = float(np.percentile(arr, 25))
            p50 = float(np.percentile(arr, 50))
            p75 = float(np.percentile(arr, 75))
            p90 = float(np.percentile(arr, 90))
            p95 = float(np.percentile(arr, 95))
        else:
            mean_len = median_len = std_len = p10 = p25 = p50 = p75 = p90 = p95 = 0.0
            min_len = max_len = 0

        # Length Buckets (<20, 20-50, 50-100, 100-200, 200-512, >512 words)
        length_buckets = {
            "<20_words": sum(1 for w in doc_word_counts if w < 20),
            "20_50_words": sum(1 for w in doc_word_counts if 20 <= w < 50),
            "50_100_words": sum(1 for w in doc_word_counts if 50 <= w < 100),
            "100_200_words": sum(1 for w in doc_word_counts if 100 <= w < 200),
            "200_512_words": sum(1 for w in doc_word_counts if 200 <= w <= 512),
            ">512_words": sum(1 for w in doc_word_counts if w > 512),
        }

        manifest = {
            "manifest_version": "2.0",
            "corpus_file": str(self.corpus_file.name),
            "corpus_path": str(self.corpus_file.resolve()),
            "file_size_bytes": file_size_bytes,
            "sha256": sha256_hash,
            "format": {
                "encoding": "UTF-8",
                "document_separator": "blank_line",
                "line_normalization": "non_empty_lines_joined_with_single_space",
                "word_tokenization": "whitespace"
            },
            "total_documents": total_docs,
            "total_words": total_words,
            "total_characters": total_chars,
            "unique_vocabulary": unique_vocab,
            "length_statistics": {
                "mean_words": round(mean_len, 2),
                "median_words": round(median_len, 2),
                "std_words": round(std_len, 2),
                "min_words": min_len,
                "max_words": max_len,
                "p10": round(p10, 2),
                "p25": round(p25, 2),
                "p50": round(p50, 2),
                "p75": round(p75, 2),
                "p90": round(p90, 2),
                "p95": round(p95, 2),
            },
            "length_buckets": length_buckets,
            "corpus_integrity": {
                "exact_duplicate_count": exact_duplicates,
                "exact_duplicate_rate": round(exact_duplicate_rate, 6)
            },
            "quality_caveats": None
        }

        return manifest

    def create_manifest(self, output_path: str) -> Dict[str, Any]:
        manifest = self.analyze_corpus()
        save_json(manifest, output_path)
        logger.info(f"Corpus manifest saved to: {output_path}")
        return manifest
