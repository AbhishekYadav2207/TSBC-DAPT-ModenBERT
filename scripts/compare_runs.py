import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
from dapt.src.utils import resolve_path, load_json, save_json, setup_logging

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description="Compare ModernBERT Baseline vs MaritimeBERT-v1 DAPT Runs")
    parser.add_argument("--baseline-dir", type=str, default="dapt/outputs/experiments/baseline-modernbert", help="Baseline experiment directory")
    parser.add_argument("--dapt-dir", type=str, default="dapt/outputs/experiments/MaritimeBERT-v1", help="DAPT experiment directory")
    parser.add_argument("--output-dir", type=str, default="dapt/outputs/experiments", help="Summary output directory")
    args = parser.parse_args()

    base_path = resolve_path(args.baseline_dir) / "evaluation_metrics.json"
    dapt_path = resolve_path(args.dapt_dir) / "evaluation_metrics.json"

    if not base_path.exists():
        logger.error(f"Baseline metrics file not found: {base_path}")
        logger.info("Run `python dapt/scripts/evaluate_dapt.py --is-baseline` first.")
        return

    if not dapt_path.exists():
        logger.error(f"DAPT metrics file not found: {dapt_path}")
        logger.info("Run `python dapt/scripts/evaluate_dapt.py` or training first.")
        return

    base_metrics = load_json(base_path)
    dapt_metrics = load_json(dapt_path)

    base_loss = base_metrics.get("mlm_loss", 0.0)
    dapt_loss = dapt_metrics.get("mlm_loss", 0.0)
    delta_loss = base_loss - dapt_loss

    base_ppl = base_metrics.get("perplexity", 0.0)
    dapt_ppl = dapt_metrics.get("perplexity", 0.0)
    
    if isinstance(base_ppl, (int, float)) and isinstance(dapt_ppl, (int, float)):
        delta_ppl = base_ppl - dapt_ppl
    else:
        delta_ppl = "N/A"

    comparison = {
        "baseline_model": base_metrics.get("model_evaluated", "ModernBERT"),
        "dapt_model": dapt_metrics.get("model_evaluated", "MaritimeBERT-v1"),
        "eval_split": base_metrics.get("eval_split", "validation"),
        "metrics": {
            "baseline_mlm_loss": base_loss,
            "dapt_mlm_loss": dapt_loss,
            "delta_mlm_loss": round(delta_loss, 4),
            "baseline_perplexity": base_ppl,
            "dapt_perplexity": dapt_ppl,
            "delta_perplexity": round(delta_ppl, 4) if isinstance(delta_ppl, float) else delta_ppl,
        }
    }

    out_dir = resolve_path(args.output_dir)
    save_json(comparison, out_dir / "comparison_report.json")

    logger.info("=== Baseline vs MaritimeBERT-v1 Evaluation Comparison ===")
    logger.info(f"Evaluation Split: {comparison['eval_split']}")
    logger.info(f"Baseline MLM Loss:   {base_loss:.4f}")
    logger.info(f"MaritimeBERT Loss:   {dapt_loss:.4f}")
    logger.info(f"Delta Loss (Gain):   {delta_loss:+.4f}")
    logger.info(f"Baseline Perplexity: {base_ppl}")
    logger.info(f"MaritimeBERT PPL:    {dapt_ppl}")
    logger.info(f"Comparison report saved to: {out_dir / 'comparison_report.json'}")

if __name__ == "__main__":
    main()
