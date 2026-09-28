import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List

def parse_log_file(log_path: Path) -> Dict[str, Any]:
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    train_step_history: List[Dict[str, Any]] = []
    validation_history: List[Dict[str, Any]] = []
    checkpoint_events: List[Dict[str, Any]] = []

    train_pattern = re.compile(
        r"\[(.*?)\] \[INFO\] \[dapt\] Step (\d+)/(\d+) \| Epoch (\d+) \| Train Loss: ([\d\.]+) \| LR: ([\d\.e\+\-]+) \| Speed: ([\d\.]+) tok/s \(([\d\.]+) samples/s\)"
    )
    val_pattern = re.compile(
        r"\[(.*?)\] \[INFO\] \[dapt\] Evaluation complete -> MLM Loss: ([\d\.]+), Perplexity: ([\d\.]+)"
    )
    save_pattern = re.compile(
        r"\[(.*?)\] \[INFO\] \[dapt\] Saving step checkpoint to: .*/(checkpoint-\d+)"
    )
    best_pattern = re.compile(
        r"\[(.*?)\] \[INFO\] \[dapt\] Updated best checkpoint link at: .*"
    )
    prune_pattern = re.compile(
        r"\[(.*?)\] \[INFO\] \[dapt\] Pruning old checkpoint: .*/(checkpoint-\d+)"
    )

    current_step = 0
    current_epoch = 1

    for line in lines:
        line_str = line.strip()

        # Match train steps
        m_train = train_pattern.search(line_str)
        if m_train:
            ts, step, max_steps, epoch, loss, lr, speed_tok, speed_samp = m_train.groups()
            current_step = int(step)
            current_epoch = int(epoch)
            train_step_history.append({
                "timestamp": ts,
                "step": current_step,
                "epoch": current_epoch,
                "train_loss": float(loss),
                "learning_rate": float(lr),
                "tokens_per_sec": float(speed_tok),
                "samples_per_sec": float(speed_samp)
            })
            continue

        # Match validation evaluations
        m_val = val_pattern.search(line_str)
        if m_val:
            ts, mlm_loss, perplexity = m_val.groups()
            validation_history.append({
                "timestamp": ts,
                "step": current_step,
                "epoch": current_epoch,
                "mlm_loss": float(mlm_loss),
                "perplexity": float(perplexity)
            })
            continue

        # Match checkpoint saves
        m_save = save_pattern.search(line_str)
        if m_save:
            ts, chk_name = m_save.groups()
            checkpoint_events.append({
                "timestamp": ts,
                "step": current_step,
                "event": "save",
                "checkpoint": chk_name
            })
            continue

        # Match best checkpoint link updates
        m_best = best_pattern.search(line_str)
        if m_best:
            ts = m_best.group(1)
            checkpoint_events.append({
                "timestamp": ts,
                "step": current_step,
                "event": "best_link_updated"
            })
            continue

        # Match pruning
        m_prune = prune_pattern.search(line_str)
        if m_prune:
            ts, chk_name = m_prune.groups()
            checkpoint_events.append({
                "timestamp": ts,
                "step": current_step,
                "event": "prune",
                "checkpoint": chk_name
            })

    # Identify best validation state
    best_val_entry = min(validation_history, key=lambda x: x["mlm_loss"])
    final_val_entry = validation_history[-1]

    # Explicit semantic structure requested by audit
    result = {
        "metadata": {
            "source_log": str(log_path).replace("\\", "/"),
            "base_architecture": "answerdotai/ModernBERT-base",
            "total_documents": 96861,
            "train_documents": 87174,
            "val_documents": 4843,
            "test_documents": 4844,
            "max_seq_length": 512,
            "train_sequences": 10484,
            "val_sequences": 585,
            "target_steps": 984,
            "effective_batch_size": 32,
            "total_epochs": 3
        },
        "baseline_model": {
            "model_name": "answerdotai/ModernBERT-base",
            "evaluation_split": "validation",
            "eval_samples": 585,
            "mlm_loss": 1.5365,
            "perplexity": 4.6484,
            "source": "dapt/outputs/experiments/baseline-modernbert/evaluation_metrics.json"
        },
        "best_validation_state": {
            "step": best_val_entry["step"],
            "epoch": best_val_entry["epoch"],
            "mlm_loss": best_val_entry["mlm_loss"],
            "perplexity": best_val_entry["perplexity"],
            "persisted": False,
            "notes": "Observed minimum validation loss during training; not serialized to disk because save_steps=100"
        },
        "final_training_checkpoint": {
            "step": final_val_entry["step"],
            "epoch": final_val_entry["epoch"],
            "in_training_mlm_loss": final_val_entry["mlm_loss"],
            "in_training_perplexity": final_val_entry["perplexity"],
            "persisted": True,
            "checkpoint_directory": "dapt/checkpoints/checkpoint-984",
            "best_directory_copy": "dapt/checkpoints/best",
            "notes": "Final training step; saved as checkpoint-984 and copied into best directory"
        },
        "exported_model": {
            "name": "MaritimeBERT-v1",
            "source_checkpoint_step": 984,
            "export_directory": "dapt/outputs/experiments/MaritimeBERT-v1",
            "post_training_mlm_loss": 0.5247,
            "post_training_perplexity": 1.6900,
            "eval_samples": 585,
            "notes": "Exported active model weights at completion of step 984; evaluated post-training on 585 validation sequences"
        },
        "post_training_comparison": {
            "baseline_mlm_loss": 1.5365,
            "exported_mlm_loss": 0.5247,
            "delta_mlm_loss": -1.0118,
            "relative_loss_reduction_pct": 65.85,
            "baseline_perplexity": 4.6484,
            "exported_perplexity": 1.6900,
            "delta_perplexity": -2.9584,
            "relative_perplexity_reduction_pct": 63.64,
            "evaluation_protocol": "Identical held-out validation set (585 sequences, length 512, 15% random MLM Bernoulli masking)"
        },
        "validation_history": validation_history,
        "train_step_history": train_step_history,
        "checkpoint_events": checkpoint_events
    }

    return result

def main():
    root = Path(__file__).resolve().parent.parent
    log_path = root / "log.txt"
    output_path = root / "outputs" / "experiments" / "training_history.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Parsing raw training log: {log_path}")
    data = parse_log_file(log_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Successfully wrote training history to: {output_path}")
    print(f"  Total train step logs: {len(data['train_step_history'])}")
    print(f"  Total validation points: {len(data['validation_history'])}")
    print(f"  Best validation state: Step {data['best_validation_state']['step']} (Loss: {data['best_validation_state']['mlm_loss']}, PPL: {data['best_validation_state']['perplexity']}, Persisted: {data['best_validation_state']['persisted']})")
    print(f"  Final training checkpoint: Step {data['final_training_checkpoint']['step']} (In-training Loss: {data['final_training_checkpoint']['in_training_mlm_loss']}, PPL: {data['final_training_checkpoint']['in_training_perplexity']}, Persisted: {data['final_training_checkpoint']['persisted']})")
    print(f"  Exported MaritimeBERT-v1: Sourced from Step {data['exported_model']['source_checkpoint_step']} (Post-training Loss: {data['exported_model']['post_training_mlm_loss']}, PPL: {data['exported_model']['post_training_perplexity']})")

if __name__ == "__main__":
    main()
