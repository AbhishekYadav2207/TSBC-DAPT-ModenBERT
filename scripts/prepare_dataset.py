import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
from dapt.src.config import DAPTConfig
from dapt.src.corpus import CorpusLoader
from dapt.src.dataset import DatasetSplitter
from dapt.src.utils import resolve_path, setup_logging

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description="Prepare & Split Maritime Corpus")
    parser.add_argument("--config", type=str, default="dapt/configs/dapt.yaml", help="Path to dapt.yaml configuration")
    args = parser.parse_args()

    config = DAPTConfig.from_yaml(args.config)
    
    loader = CorpusLoader(config.data.corpus_path)
    documents = loader.load_documents()

    splitter = DatasetSplitter(
        train_ratio=config.data.train_split,
        val_ratio=config.data.validation_split,
        test_ratio=config.data.test_split,
        seed=config.data.split_seed
    )

    output_dir = str(resolve_path(config.data.output_dir))
    manifest = splitter.prepare_and_save(documents, output_dir)

    logger.info("=== Dataset Preparation Summary ===")
    logger.info(f"Train Documents: {manifest['document_counts']['train']:,}")
    logger.info(f"Val Documents: {manifest['document_counts']['val']:,}")
    logger.info(f"Test Documents: {manifest['document_counts']['test']:,}")
    logger.info(f"Exact Leakage (Train/Val): {manifest['leakage_diagnostics']['exact_duplicate_leakage']['train_val_overlap']}")
    logger.info(f"Exact Leakage (Train/Test): {manifest['leakage_diagnostics']['exact_duplicate_leakage']['train_test_overlap']}")
    logger.info(f"Split files written to: {output_dir}")

if __name__ == "__main__":
    main()
