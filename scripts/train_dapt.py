import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
from transformers import AutoTokenizer
from dapt.src.config import DAPTConfig
from dapt.src.corpus import CorpusLoader
from dapt.src.dataset import DatasetSplitter
from dapt.src.packing import DocumentPacker
from dapt.src.model import load_modernbert_mlm_model
from dapt.src.training import DAPTTrainer
from dapt.src.reproducibility import set_seed, get_system_environment
from dapt.src.metrics import build_experiment_manifest
from dapt.src.utils import resolve_path, save_json, setup_logging

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description="Run ModernBERT Domain-Adaptive Pretraining (DAPT)")
    parser.add_argument("--config", type=str, default="dapt/configs/dapt.yaml", help="Path to dapt.yaml configuration")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to checkpoint directory to resume from")
    parser.add_argument("--max-steps", type=int, default=None, help="Override maximum training steps (e.g., 10 for smoke test)")
    parser.add_argument("--dry-run", action="store_true", help="Run 2-step dry run smoke test without full training")
    args = parser.parse_args()

    config = DAPTConfig.from_yaml(args.config)
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
        config.training.save_steps = args.max_steps
    if args.dry_run:
        config.training.max_steps = 2
        config.training.logging_steps = 1
        config.training.save_steps = 2

    # Reproducibility
    set_seed(config.training.seed)
    system_env = get_system_environment()
    logger.info(f"System Environment -> Torch: {system_env['torch_version']}, CUDA: {system_env['cuda_available']}")

    # Corpus loading & splitting
    corpus_loader = CorpusLoader(config.data.corpus_path)
    corpus_manifest = corpus_loader.analyze_corpus()
    all_docs = corpus_loader.load_documents()

    splitter = DatasetSplitter(
        train_ratio=config.data.train_split,
        val_ratio=config.data.validation_split,
        test_ratio=config.data.test_split,
        seed=config.data.split_seed
    )
    train_docs, val_docs, test_docs = splitter.split(all_docs)

    # Tokenizer & packing
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_name_or_path,
        trust_remote_code=config.model.trust_remote_code
    )

    packer = DocumentPacker(tokenizer=tokenizer, max_seq_length=config.data.max_seq_length)
    train_samples, packing_stats = packer.pack_documents(train_docs)
    val_samples, _ = packer.pack_documents(val_docs)
    
    if args.max_steps is not None or args.dry_run:
        train_samples = train_samples[:100]
        val_samples = val_samples[:50]

    # Model loading
    model = load_modernbert_mlm_model(
        name_or_path=config.model.name_or_path,
        trust_remote_code=config.model.trust_remote_code,
        gradient_checkpointing=config.training.gradient_checkpointing,
        device=config.system.device
    )

    # Training
    trainer = DAPTTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_samples=train_samples,
        val_samples=val_samples
    )

    training_metrics = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Manifest output
    exp_dir = resolve_path(config.training.output_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_experiment_manifest(
        experiment_name="MaritimeBERT-v1",
        config_dict=config.raw_dict,
        corpus_manifest=corpus_manifest,
        tokenizer_report={"tokenizer_name_or_path": config.model.tokenizer_name_or_path, "vocab_size": len(tokenizer), "boundary_token_description": str(packer.boundary_token_id)},
        system_env=system_env,
        eval_metrics={},
        training_metrics=training_metrics
    )

    save_json(manifest, exp_dir / "experiment_manifest.json")
    save_json(training_metrics, exp_dir / "training_metrics.json")
    logger.info(f"DAPT training completed successfully. Artifacts exported to {exp_dir}")

if __name__ == "__main__":
    main()
