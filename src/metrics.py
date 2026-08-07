from typing import Dict, Any, Optional
from pathlib import Path
from dapt.src.utils import save_json, load_json, setup_logging

logger = setup_logging()

def build_experiment_manifest(
    experiment_name: str,
    config_dict: Dict[str, Any],
    corpus_manifest: Dict[str, Any],
    tokenizer_report: Dict[str, Any],
    system_env: Dict[str, Any],
    eval_metrics: Dict[str, Any],
    training_metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Builds a complete, self-describing research manifest for reproducibility.
    """
    manifest = {
        "experiment_name": experiment_name,
        "model": config_dict.get("model", {}),
        "corpus": {
            "file": corpus_manifest.get("corpus_file"),
            "sha256": corpus_manifest.get("sha256"),
            "documents": corpus_manifest.get("total_documents"),
            "words": corpus_manifest.get("total_words"),
            "characters": corpus_manifest.get("total_characters"),
            "quality_caveats": corpus_manifest.get("quality_caveats")
        },
        "tokenizer": {
            "name": tokenizer_report.get("tokenizer_name_or_path"),
            "vocab_size": tokenizer_report.get("vocab_size"),
            "boundary_token": tokenizer_report.get("boundary_token_description"),
            "fertility_subwords_per_word": tokenizer_report.get("tokenizer_fertility_subwords_per_word"),
            "unk_rate": tokenizer_report.get("unk_rate"),
        },
        "hyperparameters": {
            "max_seq_length": config_dict.get("data", {}).get("max_seq_length"),
            "packing": config_dict.get("data", {}).get("packing"),
            "mlm_probability": config_dict.get("mlm", {}).get("mlm_probability"),
            "learning_rate": config_dict.get("training", {}).get("learning_rate"),
            "batch_size": config_dict.get("training", {}).get("per_device_train_batch_size"),
            "gradient_accumulation_steps": config_dict.get("training", {}).get("gradient_accumulation_steps"),
            "epochs": config_dict.get("training", {}).get("num_train_epochs"),
            "seed": config_dict.get("training", {}).get("seed"),
        },
        "system_environment": system_env,
        "evaluation_metrics": eval_metrics,
    }

    if training_metrics:
        manifest["training_metrics"] = training_metrics

    return manifest
