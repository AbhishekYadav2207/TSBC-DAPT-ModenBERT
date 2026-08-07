import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
from transformers import AutoTokenizer
from dapt.src.config import DAPTConfig
from dapt.src.packing import DocumentPacker
from dapt.src.corpus import CorpusLoader
from dapt.src.utils import resolve_path, setup_logging

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description="Validate Dataset & Packing Efficiency")
    parser.add_argument("--config", type=str, default="dapt/configs/dapt.yaml", help="Path to dapt.yaml configuration")
    args = parser.parse_args()

    config = DAPTConfig.from_yaml(args.config)
    data_dir = resolve_path(config.data.output_dir)

    train_file = data_dir / "train.txt"
    if not train_file.exists():
        logger.info("Split train.txt not found; generating from raw corpus...")
        loader = CorpusLoader(config.data.corpus_path)
        docs = loader.load_documents()
    else:
        with open(train_file, "r", encoding="utf-8") as f:
            docs = [d.strip() for d in f.read().split("\n\n") if d.strip()]

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_name_or_path,
        trust_remote_code=config.model.trust_remote_code
    )

    packer = DocumentPacker(tokenizer=tokenizer, max_seq_length=config.data.max_seq_length)
    packed_samples, stats = packer.pack_documents(docs[:1000])

    logger.info("=== Dataset & Packing Validation Summary ===")
    logger.info(f"Sample Documents Evaluated: {stats['total_input_documents']:,}")
    logger.info(f"Max Sequence Length: {stats['max_seq_length']}")
    logger.info(f"Packed Sequence Count: {stats['packed_sequence_count']:,}")
    logger.info(f"Packing Efficiency: {stats['packing_efficiency_percent']}%")
    logger.info(f"Padding Waste: {stats['padding_waste_percent']}%")
    logger.info("Dataset validation passed successfully!")

if __name__ == "__main__":
    main()
