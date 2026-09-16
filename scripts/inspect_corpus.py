import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
from dapt.src.config import DAPTConfig
from dapt.src.corpus import CorpusLoader
from dapt.src.utils import resolve_path, setup_logging

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description="Inspect Maritime Corpus & Generate SHA-256 Manifest")
    parser.add_argument("--config", type=str, default="dapt/configs/dapt.yaml", help="Path to dapt.yaml configuration")
    args = parser.parse_args()

    config = DAPTConfig.from_yaml(args.config)
    corpus_path = resolve_path(config.data.corpus_path)

    loader = CorpusLoader(str(corpus_path))
    output_manifest_path = resolve_path(config.data.output_dir) / "corpus_manifest.json"
    
    manifest = loader.create_manifest(str(output_manifest_path))

    logger.info("=== Corpus Inspection Summary ===")
    logger.info(f"Corpus File: {manifest['corpus_file']}")
    logger.info(f"Absolute Corpus Path: {manifest['corpus_path']}")
    logger.info(f"SHA-256: {manifest['sha256']}")
    logger.info(f"File Size: {manifest['file_size_bytes']:,} bytes")
    logger.info(f"Total Documents: {manifest['total_documents']:,}")
    logger.info(f"Total Words: {manifest['total_words']:,}")
    logger.info(f"Total Characters: {manifest['total_characters']:,}")
    logger.info(f"Unique Vocabulary: {manifest['unique_vocabulary']:,}")
    logger.info(f"Mean Document Length: {manifest['length_statistics']['mean_words']} words")
    logger.info(f"Median Document Length: {manifest['length_statistics']['median_words']} words")
    logger.info(f"Standard Deviation: {manifest['length_statistics']['std_words']} words")
    logger.info(f"Min/Max Document Length: {manifest['length_statistics']['min_words']} / {manifest['length_statistics']['max_words']} words")
    logger.info(f"Exact Duplicate Count: {manifest['corpus_integrity']['exact_duplicate_count']:,}")
    logger.info(f"Exact Duplicate Rate: {manifest['corpus_integrity']['exact_duplicate_rate'] * 100:.4f}%")
    logger.info(f"Manifest Path: {output_manifest_path.resolve()}")

if __name__ == "__main__":
    main()
