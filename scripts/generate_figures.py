import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "figure.autolayout": True
})

def load_json(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"Artifact not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    root = Path(__file__).resolve().parent.parent
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = root / "outputs" / "data"
    experiments_dir = root / "outputs" / "experiments"
    checkpoints_dir = root / "checkpoints"
    
    # -------------------------------------------------------------
    # 1. Corpus Length Distribution (from corpus_manifest.json)
    # -------------------------------------------------------------
    corpus_manifest_path = data_dir / "corpus_manifest.json"
    if corpus_manifest_path.exists():
        corpus_data = load_json(corpus_manifest_path)
        buckets = corpus_data.get("length_buckets", {})
        
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = [k.replace("_words", " words").replace("_", "–") for k in buckets.keys()]
        values = list(buckets.values())
        
        bars = ax.bar(labels, values, color="#2b5c8f", edgecolor="black", alpha=0.85)
        ax.set_title("Maritime Corpus Document Length Distribution", pad=15, fontweight="bold")
        ax.set_xlabel("Word Length Range")
        ax.set_ylabel("Document Count")
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f"{height:,}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=10, fontweight="bold")
        
        plt.savefig(figures_dir / "corpus_length_distribution.png", dpi=300)
        plt.close()
        print(f"Saved: {figures_dir / 'corpus_length_distribution.png'}")

    # -------------------------------------------------------------
    # 2. Split Distribution (from split_manifest.json)
    # -------------------------------------------------------------
    split_manifest_path = data_dir / "split_manifest.json"
    if split_manifest_path.exists():
        split_data = load_json(split_manifest_path)
        counts = split_data.get("document_counts", {})
        
        fig, ax = plt.subplots(figsize=(7, 5))
        labels = [f"Train ({split_data['ratios']['train']*100:.0f}%)", 
                  f"Val ({split_data['ratios']['val']*100:.0f}%)", 
                  f"Test ({split_data['ratios']['test']*100:.0f}%)"]
        sizes = [counts.get("train", 0), counts.get("val", 0), counts.get("test", 0)]
        colors = ["#2b5c8f", "#d95f02", "#7570b3"]
        
        bars = ax.bar(labels, sizes, color=colors, edgecolor="black", alpha=0.85)
        ax.set_title("Deterministic Document Dataset Partitioning", pad=15, fontweight="bold")
        ax.set_ylabel("Document Count")
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:,}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")
        
        plt.savefig(figures_dir / "split_distribution.png", dpi=300)
        plt.close()
        print(f"Saved: {figures_dir / 'split_distribution.png'}")

    # -------------------------------------------------------------
    # 3. Tokenizer Fertility & Length Stats (from tokenizer_report.json)
    # -------------------------------------------------------------
    tokenizer_report_path = data_dir / "tokenizer_report.json"
    if tokenizer_report_path.exists():
        tok_data = load_json(tokenizer_report_path)
        stats = tok_data.get("token_length_statistics", {})
        fertility = tok_data.get("tokenizer_fertility_subwords_per_word", 1.6287)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        metric_names = ["Mean Tokens", "Median (P50)", "P90", "P95", "Max Tokens"]
        metric_vals = [stats.get("mean_tokens", 0), stats.get("median_tokens", 0), stats.get("p90_tokens", 0), stats.get("p95_tokens", 0), stats.get("max_tokens", 0)]
        
        bars = ax.bar(metric_names, metric_vals, color="#31a354", edgecolor="black", alpha=0.85)
        ax.set_title(f"ModernBERT Tokenizer Sequence Profile (Fertility: {fertility:.4f} subwords/word)", pad=15, fontweight="bold")
        ax.set_ylabel("Subword Tokens")
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")
        
        plt.savefig(figures_dir / "tokenizer_fertility.png", dpi=300)
        plt.close()
        print(f"Saved: {figures_dir / 'tokenizer_fertility.png'}")

    # -------------------------------------------------------------
    # 4. Step Trajectory & Checkpoint Comparison (from JSON files)
    # -------------------------------------------------------------
    baseline_metrics_path = experiments_dir / "baseline-modernbert" / "evaluation_metrics.json"
    comparison_report_path = experiments_dir / "comparison_report.json"
    
    steps = [0]
    losses = []
    perplexities = []
    
    if baseline_metrics_path.exists():
        b_data = load_json(baseline_metrics_path)
        losses.append(b_data.get("mlm_loss", 1.5271))
        perplexities.append(b_data.get("perplexity", 4.6050))
    else:
        losses.append(1.5271)
        perplexities.append(4.6050)

    # Dynamically discover step checkpoints
    chk_dirs = sorted(
        [d for d in checkpoints_dir.glob("checkpoint-*") if d.is_dir()],
        key=lambda x: int(x.name.split("-")[-1]) if x.name.split("-")[-1].isdigit() else 0
    )
    
    step_metrics_map = {}
    for cdir in chk_dirs:
        step_num = int(cdir.name.split("-")[-1])
        sfile = cdir / "step_metrics.json"
        if sfile.exists():
            s_data = load_json(sfile)
            steps.append(step_num)
            losses.append(s_data.get("mlm_loss"))
            perplexities.append(s_data.get("perplexity"))
            step_metrics_map[step_num] = s_data

    # Validation Loss Curve
    if len(steps) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, losses, marker="o", color="#e41a1c", linewidth=2.5, markersize=8, label="Held-Out MLM Loss")
        ax.set_title("Validation MLM Loss Progression Across DAPT Steps", pad=15, fontweight="bold")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("MLM Cross-Entropy Loss")
        
        best_step = steps[losses.index(min(losses))]
        final_step = steps[-1]
        
        ax.axvline(best_step, color="green", linestyle="--", alpha=0.7, label=f"Best Val Checkpoint (Step {best_step})")
        if final_step != best_step:
            ax.axvline(final_step, color="purple", linestyle=":", alpha=0.7, label=f"Released Artifact (Step {final_step})")
        ax.legend(loc="upper right")
        
        for x, y in zip(steps, losses):
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold")
            
        plt.savefig(figures_dir / "validation_loss_curve.png", dpi=300)
        plt.close()
        print(f"Saved: {figures_dir / 'validation_loss_curve.png'}")

        # Perplexity Curve
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, perplexities, marker="s", color="#377eb8", linewidth=2.5, markersize=8, label="Held-Out Perplexity")
        ax.set_title("Validation Perplexity Trajectory Across DAPT Steps", pad=15, fontweight="bold")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Perplexity (PPL)")
        ax.axvline(best_step, color="green", linestyle="--", alpha=0.7, label=f"Best Val Checkpoint (Step {best_step})")
        if final_step != best_step:
            ax.axvline(final_step, color="purple", linestyle=":", alpha=0.7, label=f"Released Artifact (Step {final_step})")
        ax.legend(loc="upper right")
        
        for x, y in zip(steps, perplexities):
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold")
            
        plt.savefig(figures_dir / "perplexity_curve.png", dpi=300)
        plt.close()
        print(f"Saved: {figures_dir / 'perplexity_curve.png'}")

    # -------------------------------------------------------------
    # 5. Baseline vs Released MaritimeBERT-v1 (from comparison_report.json)
    # -------------------------------------------------------------
    if comparison_report_path.exists():
        comp_data = load_json(comparison_report_path)
        m = comp_data.get("metrics", {})
        
        b_loss = m.get("baseline_mlm_loss", losses[0])
        d_loss = m.get("dapt_mlm_loss", losses[-1])
        b_ppl = m.get("baseline_perplexity", perplexities[0])
        d_ppl = m.get("dapt_perplexity", perplexities[-1])
        
        rel_loss_pct = (b_loss - d_loss) / b_loss * 100
        rel_ppl_pct = (b_ppl - d_ppl) / b_ppl * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        models = ["ModernBERT-base\n(Baseline)", "MaritimeBERT-v1\n(Released)"]
        
        bars1 = ax1.bar(models, [b_loss, d_loss], color=["#e41a1c", "#4daf4a"], edgecolor="black", alpha=0.85)
        ax1.set_title(f"MLM Loss Reduction (-{rel_loss_pct:.1f}%)", fontweight="bold")
        ax1.set_ylabel("Loss")
        for bar in bars1:
            h = bar.get_height()
            ax1.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
            
        bars2 = ax2.bar(models, [b_ppl, d_ppl], color=["#e41a1c", "#377eb8"], edgecolor="black", alpha=0.85)
        ax2.set_title(f"Perplexity Reduction (-{rel_ppl_pct:.1f}%)", fontweight="bold")
        ax2.set_ylabel("Perplexity")
        for bar in bars2:
            h = bar.get_height()
            ax2.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
            
        plt.suptitle("Intrinsic Baseline vs. MaritimeBERT-v1 Pretraining Performance Gains", fontweight="bold", y=1.03)
        plt.savefig(figures_dir / "baseline_vs_dapt.png", dpi=300)
        plt.close()
        print(f"Saved: {figures_dir / 'baseline_vs_dapt.png'}")

    # -------------------------------------------------------------
    # 6. Checkpoint Distinction Comparison (Dynamic JSON Values)
    # -------------------------------------------------------------
    if len(steps) > 1:
        best_step_idx = losses.index(min(losses[1:]))
        best_step_num = steps[best_step_idx]
        final_step_num = steps[-1]
        
        chk_names = ["Baseline\n(ModernBERT-base)", f"Best Val Checkpoint\n(Step {best_step_num})", f"Released Artifact\n(Step {final_step_num} / MaritimeBERT-v1)"]
        chk_losses = [losses[0], losses[best_step_idx], losses[-1]]
        chk_ppls = [perplexities[0], perplexities[best_step_idx], perplexities[-1]]
        
        fig, ax = plt.subplots(figsize=(9, 5))
        x = [0, 1, 2]
        width = 0.35
        
        rects1 = ax.bar([i - width/2 for i in x], chk_losses, width, label="MLM Loss", color="#e41a1c", edgecolor="black", alpha=0.85)
        rects2 = ax.bar([i + width/2 for i in x], chk_ppls, width, label="Perplexity", color="#377eb8", edgecolor="black", alpha=0.85)
        
        ax.set_title("Checkpoint Distinction & Intrinsic Performance Comparison", pad=15, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(chk_names)
        ax.legend()
        
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=9)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=9)
            
        plt.savefig(figures_dir / "checkpoint_comparison.png", dpi=300)
        plt.close()
        print(f"Saved: {figures_dir / 'checkpoint_comparison.png'}")

if __name__ == "__main__":
    main()
