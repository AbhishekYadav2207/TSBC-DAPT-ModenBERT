import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
from dapt.src.config import DAPTConfig
from dapt.src.corpus import CorpusLoader
from dapt.src.tokenizer import TokenizerAnalyzer
from dapt.src.utils import resolve_path, setup_logging

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description="Analyze Tokenizer Compatibility & Diagnostics")
    parser.add_argument("--config", type=str, default="dapt/configs/dapt.yaml", help="Path to dapt.yaml configuration")
    args = parser.parse_args()

    config = DAPTConfig.from_yaml(args.config)
    
    loader = CorpusLoader(config.data.corpus_path)
    documents = loader.load_documents()

    analyzer = TokenizerAnalyzer(
        tokenizer_name_or_path=config.model.tokenizer_name_or_path,
        trust_remote_code=config.model.trust_remote_code
    )

    output_path = str(resolve_path(config.data.output_dir) / "tokenizer_report.json")
    report = analyzer.create_report(documents, output_path)

    logger.info("=== Tokenizer Diagnostic Summary ===")
    logger.info(f"Model/Tokenizer: {report['tokenizer_name_or_path']}")
    logger.info(f"Vocab Size: {report['vocab_size']:,}")
    logger.info(f"Resolved Boundary Token: {report['boundary_token_description']}")
    logger.info(f"Subword Fertility: {report['tokenizer_fertility_subwords_per_word']:.4f} subwords/word")
    logger.info(f"Mean Token Length: {report['token_length_statistics']['mean_tokens']} tokens")
    logger.info(f"Truncation Rate (>512 tokens): {report['truncation_rates']['exceeds_512_rate']*100:.2f}%")

if __name__ == "__main__":
    main()
