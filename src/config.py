import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from dapt.src.utils import resolve_path

@dataclass
class ModelConfig:
    name_or_path: str = "answerdotai/ModernBERT-base"
    tokenizer_name_or_path: Optional[str] = None
    trust_remote_code: bool = False

    def __post_init__(self):
        if not self.tokenizer_name_or_path:
            self.tokenizer_name_or_path = self.name_or_path

@dataclass
class DataConfig:
    corpus_path: str = "outputs/maritime_corpus.txt"
    output_dir: str = "dapt/outputs/data"
    train_split: float = 0.90
    validation_split: float = 0.05
    test_split: float = 0.05
    split_seed: int = 42
    max_seq_length: int = 512
    packing: bool = True
    deduplicate: bool = False
    shuffle: bool = True

    def validate(self):
        total_split = self.train_split + self.validation_split + self.test_split
        if abs(total_split - 1.0) > 1e-4:
            raise ValueError(f"Train/Val/Test splits must sum to 1.0, got {total_split:.4f}")

@dataclass
class TrainingConfig:
    output_dir: str = "dapt/outputs/experiments/MaritimeBERT-v1"
    checkpoints_dir: str = "dapt/checkpoints"
    seed: int = 42
    num_train_epochs: int = 3
    max_steps: int = -1
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    lr_scheduler_type: str = "linear"
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    save_total_limit: int = 3

@dataclass
class MLMConfig:
    mlm_probability: float = 0.15

@dataclass
class EvaluationConfig:
    enabled: bool = True
    eval_steps: int = 50
    save_metrics: bool = True

@dataclass
class SystemConfig:
    device: str = "auto"
    num_workers: int = 2

@dataclass
class DAPTConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mlm: MLMConfig = field(default_factory=MLMConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    raw_dict: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DAPTConfig":
        resolved_yaml = resolve_path(yaml_path)
        if not resolved_yaml.exists():
            raise FileNotFoundError(f"Configuration file not found: {resolved_yaml}")
        
        with open(resolved_yaml, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        model_cfg = ModelConfig(**raw.get("model", {}))
        data_cfg = DataConfig(**raw.get("data", {}))
        data_cfg.validate()
        
        training_cfg = TrainingConfig(**raw.get("training", {}))
        mlm_cfg = MLMConfig(**raw.get("mlm", {}))
        eval_cfg = EvaluationConfig(**raw.get("evaluation", {}))
        sys_cfg = SystemConfig(**raw.get("system", {}))

        return cls(
            model=model_cfg,
            data=data_cfg,
            training=training_cfg,
            mlm=mlm_cfg,
            evaluation=eval_cfg,
            system=sys_cfg,
            raw_dict=raw
        )
