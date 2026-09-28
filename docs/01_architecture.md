# 01. Subsystem Architecture & Design Principles

The DAPT subsystem is designed as an **independent, self-contained research subsystem** located entirely within `dapt/`.

---

## 1. Absolute Independence Principles

To guarantee scientific rigor and modular code structure, DAPT enforces two foundational architectural rules:

### Zero Upstream Imports Rule
The python modules in `dapt/src/` and scripts in `dapt/scripts/` do **not** import functions, classes, or configuration objects from the parent workspace root (`scripts/` or `src/`). All utilities, config parsers, tokenization routines, and trainers are fully encapsulated within `dapt/`.

### Self-Contained Central Configuration
The entire subsystem configuration is declared in `dapt/configs/dapt.yaml`. Hyperparameters, file paths, seed parameters, batch sizes, learning rate schedules, and evaluation frequencies are governed exclusively by this YAML configuration file.

---

## 2. Directory Layout

```text
dapt/
├── configs/
│   └── dapt.yaml             # Central configuration file
├── src/                      # Modular Python source code
│   ├── config.py             # Dataclass schemas & YAML parser
│   ├── corpus.py             # SHA-256 calculator & corpus analyzer
│   ├── dataset.py            # Train/val/test splitter & leakage detector
│   ├── tokenizer.py          # ModernBERT boundary token resolver
│   ├── packing.py            # Sequence packing algorithm
│   ├── masking.py            # ModernBERT 15% MLM DataCollator
│   ├── model.py              # AutoModelForMaskedLM loader
│   ├── training.py           # DAPTTrainer loop & progress tracker
│   ├── evaluation.py         # Held-out MLMEvaluator
│   ├── metrics.py            # Experiment manifest builder
│   ├── checkpointing.py      # CheckpointManager & rotation policy
│   ├── reproducibility.py    # Seed setting & hardware tracker
│   └── utils.py              # Path resolver & JSON file helpers
├── scripts/                  # Executable CLI scripts
│   ├── inspect_corpus.py     # SHA-256 & corpus_manifest.json generator
│   ├── prepare_dataset.py    # Split generator & split_manifest.json
│   ├── tokenize_corpus.py    # Tokenizer fertility & tokenizer_report.json
│   ├── validate_dataset.py   # Packing & dataset validator
│   ├── train_dapt.py         # Main DAPT training engine
│   ├── evaluate_dapt.py      # Evaluates baseline or model checkpoint
│   ├── compare_runs.py       # Computes Delta Loss & Delta Perplexity
│   └── generate_figures.py   # Automated chart generation script
├── docs/                     # 14 technical documentation pages
├── reports/                  # 6 empirical analysis reports
├── diagrams/                 # Architectural workflow diagrams
├── figures/                  # Dynamically generated PNG and vector PDF visualization charts
├── outputs/                  # Split datasets, experiment manifests & reports
├── checkpoints/              # Checkpoint directories (Step 984 Final Checkpoint / checkpoints/best)
└── tests/                    # PyTest test suite
```

---

## 3. Data Processing Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                   FROZEN UPSTREAM INPUT                     │
│               outputs/maritime_corpus.txt                   │
└──────────────────────────────┬──────────────────────────────┘
                               │ Read-Only Ingestion
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    DAPT SUBSYSTEM CORE                      │
│                                                             │
│  1. Corpus SHA-256 Lock & Analysis (`corpus.py`)             │
│  2. Deterministic Partitioning (`dataset.py`)               │
│  3. Tokenizer Diagnostic (`tokenizer.py`)                   │
│  4. Document Packing Engine (`packing.py`)                  │
│  5. ModernBERT MLM Training Loop (`training.py`)             │
│  6. Held-Out Evaluator (`evaluation.py`)                    │
│  7. Checkpoint Manager & Rotation (`checkpointing.py`)      │
└──────────────────────────────┬──────────────────────────────┘
                               │ Export
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 MaritimeBERT-v1 RELEASE                     │
│       dapt/outputs/experiments/MaritimeBERT-v1              │
└─────────────────────────────────────────────────────────────┘
```
