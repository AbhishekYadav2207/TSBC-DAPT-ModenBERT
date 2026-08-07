import math
from typing import List, Dict, Any, Union
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from dapt.src.masking import ModernBERTMaskDataCollator
from dapt.src.utils import setup_logging, save_json

logger = setup_logging()

class MLMEvaluator:
    """
    Standardized MLM evaluator for computing held-out validation and test metrics
    (MLM Loss, Perplexity, Token Counts) across both ModernBERT baseline and MaritimeBERT-v1.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        mlm_probability: float = 0.15,
        batch_size: int = 8,
        device: str = "auto"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.batch_size = batch_size
        self.collator = ModernBERTMaskDataCollator(tokenizer, mlm_probability=mlm_probability)

        if device != "auto":
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

    def evaluate(self, packed_samples: List[Dict[str, List[int]]]) -> Dict[str, Any]:
        """
        Executes evaluation pass over packed samples.
        """
        logger.info(f"Evaluating model on {len(packed_samples)} packed evaluation samples...")
        self.model.eval()

        dataloader = DataLoader(
            packed_samples,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator
        )

        total_loss = 0.0
        total_batches = 0
        total_masked_tokens = 0
        total_eval_tokens = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss.item()
                total_loss += loss
                total_batches += 1

                # Count masked tokens evaluated
                masked_in_batch = (labels != -100).sum().item()
                tokens_in_batch = (attention_mask == 1).sum().item()
                
                total_masked_tokens += masked_in_batch
                total_eval_tokens += tokens_in_batch

        mean_loss = total_loss / total_batches if total_batches > 0 else 0.0
        try:
            perplexity = math.exp(mean_loss)
        except OverflowError:
            perplexity = float("inf")

        metrics = {
            "eval_samples": len(packed_samples),
            "eval_tokens": total_eval_tokens,
            "masked_tokens_evaluated": total_masked_tokens,
            "mlm_loss": round(mean_loss, 4),
            "perplexity": round(perplexity, 4) if perplexity != float("inf") else "inf",
            "mlm_probability": self.mlm_probability,
            "batch_size": self.batch_size
        }

        logger.info(
            f"Evaluation complete -> MLM Loss: {metrics['mlm_loss']:.4f}, "
            f"Perplexity: {metrics['perplexity']}"
        )
        return metrics

    def evaluate_and_save(
        self,
        packed_samples: List[Dict[str, List[int]]],
        output_path: str
    ) -> Dict[str, Any]:
        metrics = self.evaluate(packed_samples)
        save_json(metrics, output_path)
        logger.info(f"Saved evaluation metrics to: {output_path}")
        return metrics
