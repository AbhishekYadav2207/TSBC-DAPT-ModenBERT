import os
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from dapt.src.utils import save_json, setup_logging

logger = setup_logging()

class DatasetSplitter:
    """
    Splits corpus into reproducible train, validation, and test sets.
    Computes exact duplicate leakage (REQUIRED) and near-duplicate leakage diagnostics (DIAGNOSTIC).
    Strictly preserves input documents without modifying or cleaning text.
    """

    def __init__(
        self,
        train_ratio: float = 0.90,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        seed: int = 42
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")

    def split(self, documents: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        Deterministically shuffles and splits documents.
        """
        indexed_docs = list(documents)
        rng = random.Random(self.seed)
        rng.shuffle(indexed_docs)

        n_total = len(indexed_docs)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        train_docs = indexed_docs[:n_train]
        val_docs = indexed_docs[n_train:n_train + n_val]
        test_docs = indexed_docs[n_train + n_val:]

        logger.info(
            f"Split dataset ({n_total} total docs) -> "
            f"Train: {len(train_docs)}, Val: {len(val_docs)}, Test: {len(test_docs)}"
        )
        return train_docs, val_docs, test_docs

    def compute_leakage_diagnostics(
        self,
        train_docs: List[str],
        val_docs: List[str],
        test_docs: List[str]
    ) -> Dict[str, Any]:
        """
        Computes exact duplicate leakage count and near-duplicate leakage diagnostic between splits.
        """
        train_set = set(train_docs)
        val_set = set(val_docs)
        test_set = set(test_docs)

        # Exact duplicate leakage (REQUIRED)
        train_val_exact = len(train_set.intersection(val_set))
        train_test_exact = len(train_set.intersection(test_set))
        val_test_exact = len(val_set.intersection(test_set))

        # Near-duplicate diagnostic via 5-gram shingles (DIAGNOSTIC sample cap for speed)
        sample_cap = 2000
        val_sample = val_docs[:sample_cap]
        test_sample = test_docs[:sample_cap]

        def get_shingles(doc: str, k: int = 5) -> Set[str]:
            tokens = doc.lower().split()
            if len(tokens) < k:
                return {" ".join(tokens)}
            return {" ".join(tokens[i:i+k]) for i in range(len(tokens) - k + 1)}

        train_shingles = set()
        for doc in train_docs[:5000]:
            train_shingles.update(get_shingles(doc))

        val_near_leakage = 0
        for doc in val_sample:
            shingles = get_shingles(doc)
            if shingles and len(shingles.intersection(train_shingles)) / len(shingles) > 0.6:
                val_near_leakage += 1

        test_near_leakage = 0
        for doc in test_sample:
            shingles = get_shingles(doc)
            if shingles and len(shingles.intersection(train_shingles)) / len(shingles) > 0.6:
                test_near_leakage += 1

        return {
            "exact_duplicate_leakage": {
                "train_val_overlap": train_val_exact,
                "train_test_overlap": train_test_exact,
                "val_test_overlap": val_test_exact,
            },
            "near_duplicate_leakage_diagnostic": {
                "sample_evaluated": sample_cap,
                "val_high_shingle_overlap_count": val_near_leakage,
                "val_high_shingle_overlap_rate": round(val_near_leakage / len(val_sample), 4) if val_sample else 0.0,
                "test_high_shingle_overlap_count": test_near_leakage,
                "test_high_shingle_overlap_rate": round(test_near_leakage / len(test_sample), 4) if test_sample else 0.0,
            }
        }

    def prepare_and_save(
        self,
        documents: List[str],
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Splits documents, computes leakage diagnostics, saves split files, and produces manifest.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        train_docs, val_docs, test_docs = self.split(documents)
        leakage = self.compute_leakage_diagnostics(train_docs, val_docs, test_docs)

        # Write split files (separated by double newlines to preserve record format)
        train_file = out_path / "train.txt"
        val_file = out_path / "val.txt"
        test_file = out_path / "test.txt"

        with open(train_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(train_docs))
        with open(val_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(val_docs))
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(test_docs))

        manifest = {
            "split_seed": self.seed,
            "ratios": {
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio
            },
            "document_counts": {
                "train": len(train_docs),
                "val": len(val_docs),
                "test": len(test_docs),
                "total": len(documents)
            },
            "word_counts": {
                "train": sum(len(d.split()) for d in train_docs),
                "val": sum(len(d.split()) for d in val_docs),
                "test": sum(len(d.split()) for d in test_docs),
            },
            "file_paths": {
                "train": str(train_file.resolve()),
                "val": str(val_file.resolve()),
                "test": str(test_file.resolve()),
            },
            "leakage_diagnostics": leakage
        }

        save_json(manifest, out_path / "split_manifest.json")
        logger.info(f"Saved dataset splits and split_manifest.json to {out_path}")
        return manifest
