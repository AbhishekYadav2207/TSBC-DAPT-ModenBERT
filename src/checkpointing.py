import os
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from dapt.src.utils import setup_logging, save_json, resolve_path

logger = setup_logging()

class CheckpointManager:
    """
    Manages saving, rotation, best model selection, and reloading of DAPT checkpoints.
    Ensures final export directory (MaritimeBERT-v1) is loadable directly via standard HF APIs.
    """

    def __init__(
        self,
        checkpoints_dir: str,
        export_dir: str,
        save_total_limit: int = 3
    ):
        self.checkpoints_dir = resolve_path(checkpoints_dir)
        self.export_dir = resolve_path(export_dir)
        self.save_total_limit = save_total_limit

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")

    def save_checkpoint(
        self,
        step: int,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        metrics: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> Path:
        """
        Saves a step checkpoint directory (checkpoint-STEP) containing model, tokenizer, and metrics.
        """
        step_dir = self.checkpoints_dir / f"checkpoint-{step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving step checkpoint to: {step_dir}")
        model.save_pretrained(step_dir)
        tokenizer.save_pretrained(step_dir)

        if metrics:
            save_json(metrics, step_dir / "step_metrics.json")

        if is_best:
            best_dir = self.checkpoints_dir / "best"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(step_dir, best_dir)
            logger.info(f"Updated best checkpoint link at: {best_dir}")

        self._rotate_checkpoints()
        return step_dir

    def export_final_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        experiment_manifest: Dict[str, Any]
    ) -> Path:
        """
        Saves final MaritimeBERT-v1 export model directory with configuration and manifest.
        """
        logger.info(f"Exporting final MaritimeBERT-v1 model to: {self.export_dir}")
        model.save_pretrained(self.export_dir)
        tokenizer.save_pretrained(self.export_dir)
        save_json(experiment_manifest, self.export_dir / "experiment_manifest.json")
        return self.export_dir

    def _rotate_checkpoints(self) -> None:
        """
        Prunes older step checkpoints to enforce save_total_limit.
        """
        if self.save_total_limit <= 0:
            return

        checkpoints = sorted(
            [d for d in self.checkpoints_dir.glob("checkpoint-*") if d.is_dir()],
            key=lambda x: int(x.name.split("-")[-1]) if x.name.split("-")[-1].isdigit() else 0
        )

        while len(checkpoints) > self.save_total_limit:
            oldest = checkpoints.pop(0)
            logger.info(f"Pruning old checkpoint: {oldest}")
            shutil.rmtree(oldest)

    @classmethod
    def get_latest_checkpoint(cls, checkpoints_dir: str) -> Optional[Path]:
        """
        Finds the latest numerical checkpoint directory in checkpoints_dir.
        """
        cdir = Path(checkpoints_dir)
        if not cdir.exists():
            return None
        checkpoints = sorted(
            [d for d in cdir.glob("checkpoint-*") if d.is_dir()],
            key=lambda x: int(x.name.split("-")[-1]) if x.name.split("-")[-1].isdigit() else 0
        )
        return checkpoints[-1] if checkpoints else None
