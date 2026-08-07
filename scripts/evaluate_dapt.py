import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM
from dapt.src.config import DAPTConfig
from dapt.src.corpus import CorpusLoader
from dapt.src.dataset import DatasetSplitter
from dapt.src.packing import DocumentPacker
from dapt.src.evaluation import MLMEvaluator
from dapt.src.utils import resolve_path, save_json, setup_logging

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description="Evaluate ModernBERT Baseline or MaritimeBERT Checkpoint")
    parser.add_argument("--config", type=str, default="dapt/configs/dapt.yaml", help="Path to dapt.yaml configuration")
    parser.add_argument("--model-path", type=str, default=None, help="Explicit model or checkpoint path to evaluate")
    parser.add_argument("--eval-split", type=str, default="validation", choices=["validation", "test"], help="Dataset split to evaluate on")
    parser.add_argument("--is-baseline", action="store_true", help="Evaluate untouched ModernBERT baseline")
    args = parser.parse_args()

    config = DAPTConfig.from_yaml(args.config)

    # Determine model path & target output directory
    if args.is_baseline:
        model_name_or_path = config.model.name_or_path
        exp_dir = resolve_path("dapt/outputs/experiments/baseline-modernbert")
        logger.info(f"Evaluating untouched baseline model: {model_name_or_path}")
    else:
        exp_dir = resolve_path(config.training.output_dir)
        if args.model_path:
            resolved_model_path = resolve_path(args.model_path)
            if not resolved_model_path.exists():
                raise FileNotFoundError(f"Model or checkpoint path does not exist: {resolved_model_path}")
            model_name_or_path = str(resolved_model_path)
            logger.info(f"Evaluating custom model/checkpoint path: {model_name_or_path}")
        else:
            model_name_or_path = str(resolve_path(config.training.output_dir))
            logger.info(f"Evaluating final MaritimeBERT-v1 model: {model_name_or_path}")

    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load corpus & split
    loader = CorpusLoader(config.data.corpus_path)
    all_docs = loader.load_documents()

    splitter = DatasetSplitter(
        train_ratio=config.data.train_split,
        val_ratio=config.data.validation_split,
        test_ratio=config.data.test_split,
        seed=config.data.split_seed
    )
    train_docs, val_docs, test_docs = splitter.split(all_docs)
    eval_docs = val_docs if args.eval_split == "validation" else test_docs

    # Tokenizer & packing
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_name_or_path,
        trust_remote_code=config.model.trust_remote_code
    )
    packer = DocumentPacker(tokenizer=tokenizer, max_seq_length=config.data.max_seq_length)
    eval_samples, packing_stats = packer.pack_documents(eval_docs)

    # Model loading
    model = AutoModelForMaskedLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=config.model.trust_remote_code
    )

    # Evaluate
    evaluator = MLMEvaluator(
        model=model,
        tokenizer=tokenizer,
        mlm_probability=config.mlm.mlm_probability,
        batch_size=config.training.per_device_eval_batch_size,
        device=config.system.device
    )

    metrics = evaluator.evaluate(eval_samples)
    metrics["eval_split"] = args.eval_split
    metrics["model_evaluated"] = model_name_or_path
    metrics["is_baseline"] = args.is_baseline

    save_json(metrics, exp_dir / "evaluation_metrics.json")
    logger.info(f"Evaluation complete. Metrics saved to: {exp_dir / 'evaluation_metrics.json'}")

if __name__ == "__main__":
    main()
