import time
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_scheduler
)
from dapt.src.config import DAPTConfig
from dapt.src.masking import ModernBERTMaskDataCollator
from dapt.src.evaluation import MLMEvaluator
from dapt.src.checkpointing import CheckpointManager
from dapt.src.utils import setup_logging, save_json

logger = setup_logging()

class DAPTTrainer:
    """
    Independent DAPT training engine for ModernBERT.
    Handles step-based training, mixed-precision, gradient accumulation, periodic evaluation,
    checkpointing, resume capability, and throughput metrics logging.
    """

    def __init__(
        self,
        config: DAPTConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_samples: List[Dict[str, List[int]]],
        val_samples: Optional[List[Dict[str, List[int]]]] = None
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_samples = train_samples
        self.val_samples = val_samples

        self.t_cfg = config.training
        self.m_cfg = config.mlm
        self.s_cfg = config.system

        # Device assignment
        if self.s_cfg.device != "auto":
            self.device = torch.device(self.s_cfg.device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

        # Data collator and dataloader
        self.collator = ModernBERTMaskDataCollator(tokenizer, mlm_probability=self.m_cfg.mlm_probability)
        self.train_dataloader = DataLoader(
            self.train_samples,
            batch_size=self.t_cfg.per_device_train_batch_size,
            shuffle=self.data_shuffle_flag(),
            collate_fn=self.collator
        )

        # Checkpoint Manager
        self.checkpoint_mgr = CheckpointManager(
            checkpoints_dir=self.t_cfg.checkpoints_dir,
            export_dir=self.t_cfg.output_dir,
            save_total_limit=self.t_cfg.save_total_limit
        )

        # Evaluator
        self.evaluator = None
        if self.val_samples and config.evaluation.enabled:
            self.evaluator = MLMEvaluator(
                model=self.model,
                tokenizer=self.tokenizer,
                mlm_probability=self.m_cfg.mlm_probability,
                batch_size=self.t_cfg.per_device_eval_batch_size,
                device=str(self.device)
            )

    def data_shuffle_flag(self) -> bool:
        return self.config.data.shuffle

    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes DAPT training loop.
        """
        start_step = 0
        if resume_from_checkpoint:
            resume_path = Path(resume_from_checkpoint)
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume checkpoint directory not found: {resume_path}")
            logger.info(f"Resuming model training from checkpoint: {resume_path}")
            self.model = self.model.from_pretrained(resume_path).to(self.device)
            step_str = resume_path.name.split("-")[-1]
            if step_str.isdigit():
                start_step = int(step_str)

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.t_cfg.learning_rate,
            weight_decay=self.t_cfg.weight_decay
        )

        total_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.t_cfg.gradient_accumulation_steps)
        max_train_steps = self.t_cfg.max_steps
        if max_train_steps <= 0:
            max_train_steps = total_steps_per_epoch * self.t_cfg.num_train_epochs
            num_epochs = self.t_cfg.num_train_epochs
        else:
            num_epochs = math.ceil(max_train_steps / max(1, total_steps_per_epoch))

        warmup_steps = int(max_train_steps * self.t_cfg.warmup_ratio)
        lr_scheduler = get_scheduler(
            name=self.t_cfg.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps
        )

        logger.info(
            f"Starting DAPT Training -> Target Steps: {max_train_steps}, "
            f"Per-device Batch Size: {self.t_cfg.per_device_train_batch_size}, "
            f"Grad Accumulation: {self.t_cfg.gradient_accumulation_steps}, "
            f"Effective Batch Size: {self.t_cfg.per_device_train_batch_size * self.t_cfg.gradient_accumulation_steps}"
        )

        self.model.train()
        global_step = start_step
        total_tokens_processed = 0
        total_samples_processed = 0
        start_time = time.time()
        running_loss = 0.0

        step_history = []
        best_val_loss = float("inf")
        last_val_metrics = None

        for epoch in range(num_epochs):
            for step_idx, batch in enumerate(self.train_dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss / self.t_cfg.gradient_accumulation_steps
                loss.backward()
                running_loss += loss.item() * self.t_cfg.gradient_accumulation_steps

                total_tokens_processed += (attention_mask == 1).sum().item()
                total_samples_processed += input_ids.size(0)

                if (step_idx + 1) % self.t_cfg.gradient_accumulation_steps == 0 or (step_idx + 1) == len(self.train_dataloader):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Logging
                    if global_step % self.t_cfg.logging_steps == 0 or global_step == max_train_steps:
                        elapsed = time.time() - start_time
                        tok_per_sec = total_tokens_processed / elapsed if elapsed > 0 else 0.0
                        samp_per_sec = total_samples_processed / elapsed if elapsed > 0 else 0.0
                        avg_loss = running_loss / self.t_cfg.logging_steps if global_step > start_step else running_loss
                        current_lr = lr_scheduler.get_last_lr()[0]

                        logger.info(
                            f"Step {global_step}/{max_train_steps} | Epoch {epoch+1} | "
                            f"Train Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
                            f"Speed: {tok_per_sec:.1f} tok/s ({samp_per_sec:.1f} samples/s)"
                        )

                        step_history.append({
                            "step": global_step,
                            "epoch": epoch + 1,
                            "train_loss": round(avg_loss, 4),
                            "learning_rate": current_lr,
                            "tokens_per_sec": round(tok_per_sec, 2),
                            "samples_per_sec": round(samp_per_sec, 2),
                        })
                        running_loss = 0.0

                    # Evaluation
                    val_metrics = None
                    is_best = False
                    if self.evaluator and (global_step % self.t_cfg.eval_steps == 0 or global_step == max_train_steps):
                        val_metrics = self.evaluator.evaluate(self.val_samples)
                        last_val_metrics = val_metrics
                        val_loss = val_metrics["mlm_loss"]
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            is_best = True
                        self.model.train()

                    # Save Checkpoint
                    if global_step % self.t_cfg.save_steps == 0 or global_step == max_train_steps:
                        self.checkpoint_mgr.save_checkpoint(
                            step=global_step,
                            model=self.model,
                            tokenizer=self.tokenizer,
                            metrics=val_metrics,
                            is_best=is_best
                        )

                    if global_step >= max_train_steps:
                        break

            if global_step >= max_train_steps:
                break

        # Save final step checkpoint if not already saved
        self.checkpoint_mgr.save_checkpoint(
            step=global_step,
            model=self.model,
            tokenizer=self.tokenizer,
            metrics=last_val_metrics,
            is_best=True
        )

        total_elapsed = time.time() - start_time
        summary_metrics = {
            "total_steps": global_step,
            "total_epochs": epoch + 1,
            "total_training_duration_seconds": round(total_elapsed, 2),
            "total_tokens_processed": total_tokens_processed,
            "total_samples_processed": total_samples_processed,
            "average_throughput_tokens_per_sec": round(total_tokens_processed / total_elapsed, 2) if total_elapsed > 0 else 0.0,
            "best_validation_loss": round(best_val_loss, 4) if best_val_loss != float("inf") else None,
            "step_history": step_history
        }

        # Export final MaritimeBERT-v1 checkpoint directory
        self.checkpoint_mgr.export_final_model(
            model=self.model,
            tokenizer=self.tokenizer,
            experiment_manifest=summary_metrics
        )

        return summary_metrics
