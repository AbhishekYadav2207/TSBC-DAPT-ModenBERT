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
    corpus_path = config.data.corpus_path

    loader = CorpusLoader(corpus_path)
    output_manifest_path = resolve_path(config.data.output_dir) / "corpus_manifest.json"
    
    manifest = loader.create_manifest(str(output_manifest_path))

    logger.info("=== Corpus Inspection Summary ===")
    logger.info(f"File: {manifest['corpus_file']}")
    logger.info(f"SHA-256: {manifest['sha256']}")
    logger.info(f"Total Documents: {manifest['total_documents']:,}")
    logger.info(f"Total Words: {manifest['total_words']:,}")
    logger.info(f"Total Characters: {manifest['total_characters']:,}")
    logger.info(f"Mean Doc Length: {manifest['length_statistics']['mean_words']} words")
    logger.info(f"Exact Duplicate Count: {manifest['exact_duplicate_count']:,} ({manifest['exact_duplicate_rate']*100:.2f}%)")
    logger.info(f"Quality Caveat Status: {manifest['quality_caveats']['overall_pretraining_readiness']}")

if __name__ == "__main__":
    main()
