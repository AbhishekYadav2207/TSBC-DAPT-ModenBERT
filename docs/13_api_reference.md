# 13. CLI & Python Source Code API Reference

Complete reference guide for all CLI entrypoints and Python source code modules in `dapt/`.

---

## 1. CLI Executable Reference (`dapt/scripts/`)

### `inspect_corpus.py`
- **Purpose**: Computes SHA-256 hash lock, line/word/char counts, length buckets, and exports `corpus_manifest.json`.
- **CLI**: `python dapt/scripts/inspect_corpus.py --config dapt/configs/dapt.yaml`

### `prepare_dataset.py`
- **Purpose**: Splits corpus into 90/5/5 train/val/test partitions, checks exact and 3-shingle overlap leakage, exports `split_manifest.json`.
- **CLI**: `python dapt/scripts/prepare_dataset.py --config dapt/configs/dapt.yaml`

### `tokenize_corpus.py`
- **Purpose**: Resolves ModernBERT boundary token ID (`50282`), computes subword fertility (1.6287), checks truncation, exports `tokenizer_report.json`.
- **CLI**: `python dapt/scripts/tokenize_corpus.py --config dapt/configs/dapt.yaml`

### `validate_dataset.py`
- **Purpose**: Validates split line counts, file integrity, and document sequence packing efficiency.
- **CLI**: `python dapt/scripts/validate_dataset.py --config dapt/configs/dapt.yaml`

### `train_dapt.py`
- **Purpose**: Executes main DAPTTrainer loop, periodic validation, checkpoint rotation, exports `MaritimeBERT-v1`.
- **CLI**: `python dapt/scripts/train_dapt.py --config dapt/configs/dapt.yaml`
- **Arguments**: `--max-steps N`, `--resume-from-checkpoint PATH`

### `evaluate_dapt.py`
- **Purpose**: Evaluates baseline model or specified checkpoint on held-out split, computing MLM loss and perplexity.
- **CLI**: `python dapt/scripts/evaluate_dapt.py --config dapt/configs/dapt.yaml [--is-baseline] [--model-path PATH]`

### `compare_runs.py`
- **Purpose**: Compares baseline vs DAPT evaluation metrics and exports `comparison_report.json`.
- **CLI**: `python dapt/scripts/compare_runs.py`

### `generate_figures.py`
- **Purpose**: Parses JSON manifests and step metrics to generate PNG visualization charts in `dapt/figures/`.
- **CLI**: `python dapt/scripts/generate_figures.py`

### `validate_documentation.py`
- **Purpose**: Validates Markdown files, relative links, figures, diagrams, JSON artifacts, and checkpoint terminology.
- **CLI**: `python dapt/scripts/validate_documentation.py`

---

## 2. Python Source Code API Reference (`dapt/src/`)

- [config.py](file:///d:/CAIR/TSBC-Pipeline/dapt/src/config.py): Dataclasses (`DAPTConfig`, `TrainingConfig`, `DataConfig`, `MLMConfig`) and YAML loader `load_config(path: str) -> DAPTConfig`.
- [corpus.py](file:///d:/CAIR/TSBC-Pipeline/dapt/src/corpus.py): `inspect_corpus_file(corpus_path: Path) -> dict`, `compute_sha256(path: Path) -> str`.
- [dataset.py](file:///d:/CAIR/TSBC-Pipeline/dapt/src/dataset.py): `prepare_splits(config: DAPTConfig) -> dict`, `check_leakage(train_path, val_path, test_path) -> dict`.
- [tokenizer.py](file:///d:/CAIR/TSBC-Pipeline/dapt/src/tokenizer.py): `resolve_boundary_token_id(tokenizer) -> int`, `analyze_tokenizer_fertility(...) -> dict`.
- [packing.py](file:///d:/CAIR/TSBC-Pipeline/dapt/src/packing.py): `pack_tokenized_documents(tokenized_docs, max_seq_length=512, boundary_token_id=50282) -> list`.
- [masking.py](file:///d:/CAIR/TSBC-Pipeline/dapt/src/masking.py): `ModernBERTMaskDataCollator(tokenizer, mlm_probability=0.15)`.
- [training.py](file:///d:/CAIR/TSBC-Pipeline/dapt/src/training.py): `DAPTTrainer(config, model, tokenizer, train_samples, val_samples)` with `.train(resume_from_checkpoint=None)`.
- [evaluation.py](file:///d:/CAIR/TSBC-Pipeline/dapt/src/evaluation.py): `MLMEvaluator(model, tokenizer, mlm_probability, batch_size, device)` with `.evaluate(samples) -> dict`.
- [checkpointing.py](file:///d:/CAIR/TSBC-Pipeline/dapt/src/checkpointing.py): `CheckpointManager(checkpoints_dir, export_dir, save_total_limit=3)`.
