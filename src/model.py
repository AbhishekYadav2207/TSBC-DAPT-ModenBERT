import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from dapt.src.utils import setup_logging

logger = setup_logging()

def load_modernbert_mlm_model(
    name_or_path: str,
    trust_remote_code: bool = False,
    gradient_checkpointing: bool = False,
    device: str = "auto"
) -> PreTrainedModel:
    """
    Loads pretrained ModernBERT architecture for Masked Language Modeling.
    Supports local directories, Hugging Face hub checkpoints, and device placement.
    """
    logger.info(f"Loading ModernBERT MLM model from: {name_or_path}")
    model = AutoModelForMaskedLM.from_pretrained(
        name_or_path,
        trust_remote_code=trust_remote_code
    )

    if gradient_checkpointing:
        logger.info("Enabling gradient checkpointing for memory optimization.")
        model.gradient_checkpointing_enable()

    if device != "auto":
        target_device = torch.device(device)
        model.to(target_device)
        logger.info(f"Explicitly assigned model to device: {target_device}")
    elif torch.cuda.is_available():
        model.to(torch.device("cuda"))
        logger.info("Automatically placed model on CUDA GPU.")
    else:
        logger.info("CUDA unavailable; running model on CPU.")

    return model
