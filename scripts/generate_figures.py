import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
    "figure.autolayout": True
})

def load_json(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"Artifact not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_fig_dual(fig, base_dir: Path, name: str, also_dir: Path = None):
    """Saves figure in both high-resolution PNG (300 DPI) and vector PDF formats."""
    for out_dir in [base_dir, also_dir]:
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            png_path = out_dir / f"{name}.png"
            pdf_path = out_dir / f"{name}.pdf"
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"Saved: {name}.png and {name}.pdf")

def main():
    root = Path(__file__).resolve().parent.parent
    figures_dir = root / "figures"
    outputs_figures_dir = root / "outputs" / "figures"
    data_dir = root / "outputs" / "data"
    experiments_dir = root / "outputs" / "experiments"

    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs_figures_dir.mkdir(parents=True, exist_ok=True)

    history_path = experiments_dir / "training_history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing training history: {history_path}. Run parse_training_history.py first.")
    
    history = load_json(history_path)
    val_history = history["validation_history"]
    train_history = history["train_step_history"]
    best_state = history["best_validation_state"]
    final_chk = history["final_training_checkpoint"]
    exported_model = history["exported_model"]
    baseline_model = history["baseline_model"]
    comparison = history["post_training_comparison"]

    val_steps = [v["step"] for v in val_history]
    val_losses = [v["mlm_loss"] for v in val_history]
    val_ppls = [v["perplexity"] for v in val_history]

    # -------------------------------------------------------------
    # Figure 1: Validation MLM Loss Progression
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(val_steps, val_losses, marker="o", color="#1f77b4", linewidth=2.2, markersize=6, label="Validation MLM Loss")
    ax.set_title("ModernBERT DAPT Validation MLM Loss Trajectory (Current Run: 984 Steps)", pad=14, fontweight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Held-Out MLM Cross-Entropy Loss")
    ax.set_xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 984])

    # Mark Best Validation State (Step 950, unpersisted)
    ax.scatter([best_state["step"]], [best_state["mlm_loss"]], color="#2ca02c", s=140, zorder=5, edgecolors="black", linewidth=1.5)
    ax.annotate(
        f"Best Val State (Step {best_state['step']})\nLoss: {best_state['mlm_loss']:.4f}\n[Unpersisted State]",
        xy=(best_state["step"], best_state["mlm_loss"]),
        xytext=(-140, 30),
        textcoords="offset points",
        ha="center",
        fontsize=9.5,
        fontweight="bold",
        color="#1b5e20",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5)
    )

    # Mark Final Training Checkpoint (Step 984, persisted)
    ax.scatter([final_chk["step"]], [final_chk["in_training_mlm_loss"]], color="#d62728", s=140, zorder=5, edgecolors="black", linewidth=1.5)
    ax.annotate(
        f"Final Checkpoint (Step {final_chk['step']})\nLoss: {final_chk['in_training_mlm_loss']:.4f}\n[Persisted & Exported]",
        xy=(final_chk["step"], final_chk["in_training_mlm_loss"]),
        xytext=(30, 40),
        textcoords="offset points",
        ha="center",
        fontsize=9.5,
        fontweight="bold",
        color="#b71c1c",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5)
    )

    ax.axvline(best_state["step"], color="#2ca02c", linestyle="--", alpha=0.5, label="Step 950: Best Val State (Unpersisted)")
    ax.axvline(final_chk["step"], color="#d62728", linestyle=":", alpha=0.5, label="Step 984: Final Checkpoint (Persisted)")
    ax.legend(loc="upper right", frameon=True)

    save_fig_dual(fig, figures_dir, "validation_loss_curve", outputs_figures_dir)
    plt.close()

    # -------------------------------------------------------------
    # Figure 2: Validation Perplexity Trajectory
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(val_steps, val_ppls, marker="s", color="#ff7f0e", linewidth=2.2, markersize=6, label="Validation Perplexity")
    ax.set_title("ModernBERT DAPT Validation Perplexity Trajectory across Steps", pad=14, fontweight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Held-Out Perplexity (PPL)")
    ax.set_xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 984])

    ax.scatter([best_state["step"]], [best_state["perplexity"]], color="#2ca02c", s=140, zorder=5, edgecolors="black", linewidth=1.5)
    ax.annotate(
        f"Best Val State (Step {best_state['step']})\nPPL: {best_state['perplexity']:.4f}\n[Unpersisted State]",
        xy=(best_state["step"], best_state["perplexity"]),
        xytext=(-140, 30),
        textcoords="offset points",
        ha="center",
        fontsize=9.5,
        fontweight="bold",
        color="#1b5e20",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5)
    )

    ax.scatter([final_chk["step"]], [final_chk["in_training_perplexity"]], color="#d62728", s=140, zorder=5, edgecolors="black", linewidth=1.5)
    ax.annotate(
        f"Final Checkpoint (Step {final_chk['step']})\nPPL: {final_chk['in_training_perplexity']:.4f}\n[Persisted & Exported]",
        xy=(final_chk["step"], final_chk["in_training_perplexity"]),
        xytext=(30, 40),
        textcoords="offset points",
        ha="center",
        fontsize=9.5,
        fontweight="bold",
        color="#b71c1c",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5)
    )

    ax.axvline(best_state["step"], color="#2ca02c", linestyle="--", alpha=0.5, label="Step 950: Best Val State (Unpersisted)")
    ax.axvline(final_chk["step"], color="#d62728", linestyle=":", alpha=0.5, label="Step 984: Final Checkpoint (Persisted)")
    ax.legend(loc="upper right", frameon=True)

    save_fig_dual(fig, figures_dir, "perplexity_curve", outputs_figures_dir)
    plt.close()

    # -------------------------------------------------------------
    # Figure 3: Post-Training Baseline vs Exported Model Comparison
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    models = ["ModernBERT-base\n(Untouched Baseline)", "MaritimeBERT-v1\n(Exported Model)"]
    
    b_loss = baseline_model["mlm_loss"]
    d_loss = exported_model["post_training_mlm_loss"]
    b_ppl = baseline_model["perplexity"]
    d_ppl = exported_model["post_training_perplexity"]
    
    rel_loss_pct = comparison["relative_loss_reduction_pct"]
    rel_ppl_pct = comparison["relative_perplexity_reduction_pct"]

    bars1 = ax1.bar(models, [b_loss, d_loss], color=["#e41a1c", "#2ca02c"], edgecolor="black", alpha=0.85, width=0.5)
    ax1.set_title(f"Post-Training MLM Loss Reduction\n(-{rel_loss_pct:.2f}% | $\\Delta = {d_loss - b_loss:.4f}$)", fontweight="bold")
    ax1.set_ylabel("Held-Out MLM Cross-Entropy Loss")
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    bars2 = ax2.bar(models, [b_ppl, d_ppl], color=["#e41a1c", "#1f77b4"], edgecolor="black", alpha=0.85, width=0.5)
    ax2.set_title(f"Post-Training Perplexity Reduction\n(-{rel_ppl_pct:.2f}% | $\\Delta = {d_ppl - b_ppl:.4f}$)", fontweight="bold")
    ax2.set_ylabel("Held-Out Perplexity (PPL)")
    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plt.suptitle("Post-Training Baseline vs. Exported Model Evaluation (585 Packed Val Sequences)", fontweight="bold", y=1.03)
    save_fig_dual(fig, figures_dir, "baseline_vs_dapt", outputs_figures_dir)
    plt.close()

    # -------------------------------------------------------------
    # Figure 4: Learning Rate Schedule Trajectory
    # -------------------------------------------------------------
    train_steps = [t["step"] for t in train_history]
    train_lrs = [t["learning_rate"] for t in train_history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_steps, train_lrs, color="#6a3d9a", linewidth=2.4, label="Learning Rate ($\eta$)")
    ax.set_title("DAPT Learning Rate Schedule (Warmup: 59 steps, Linear Decay to Step 984)", pad=14, fontweight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Learning Rate")
    ax.set_xticks([0, 59, 200, 400, 600, 800, 950, 984])
    ax.axvline(59, color="#b15928", linestyle="--", alpha=0.7, label="Peak Warmup (Step 59: 5.0e-5)")
    ax.axvline(984, color="#e31a1c", linestyle=":", alpha=0.7, label="Terminal Step (Step 984: 0.0)")
    ax.legend(loc="upper right", frameon=True)

    save_fig_dual(fig, figures_dir, "learning_rate_trajectory", outputs_figures_dir)
    plt.close()

    # -------------------------------------------------------------
    # Figure 5: Model States & Checkpoint Lifecycle Architecture
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # 4 defined states with explicit persistence metadata
    states = [
        "1. ModernBERT-base\n(Untouched Baseline Control)\n[Persisted Pretrained]",
        "2. Step 950 Val State\n(In-Training Minimum Loss)\n[UNPERSISTED STATE]",
        "3. Step 984 Final Checkpoint\n(In-Training Terminal Step)\n[PERSISTED CHECKPOINT]",
        "4. MaritimeBERT-v1\n(Exported Model Artifact)\n[PERSISTED RELEASE]"
    ]
    
    losses = [
        b_loss,
        best_state["mlm_loss"],
        final_chk["in_training_mlm_loss"],
        exported_model["post_training_mlm_loss"]
    ]
    
    ppls = [
        b_ppl,
        best_state["perplexity"],
        final_chk["in_training_perplexity"],
        exported_model["post_training_perplexity"]
    ]
    
    x = [0, 1, 2, 3]
    width = 0.35

    rects1 = ax.bar([i - width/2 for i in x], losses, width, label="MLM Loss", color=["#e41a1c", "#2ca02c", "#ff7f0e", "#1f77b4"], edgecolor="black", alpha=0.85)
    rects2 = ax.bar([i + width/2 for i in x], ppls, width, label="Perplexity", color=["#fbb4ae", "#b3cde3", "#ccebc5", "#decbe4"], edgecolor="black", alpha=0.85)

    # Style hatches on Step 950 to highlight non-persistence
    rects1[1].set_hatch("//")
    rects2[1].set_hatch("//")

    ax.set_title("Model State & Checkpoint Distinction (Explicit Persistence Semantics)", pad=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(states, fontsize=9.5)
    ax.set_ylabel("Metric Value")
    ax.legend(loc="upper right", frameon=True)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=9)

    # Add explanatory footnote inside plot
    ax.text(0.5, -0.22, "* Note: Step 950 (hatched) was the empirical loss minimum during training but was not saved to disk (save_steps=100).\nMaritimeBERT-v1 weights originate from the Step 984 final persisted checkpoint.",
            ha="center", va="top", transform=ax.transAxes, fontsize=9.5, style="italic", color="#333333")

    save_fig_dual(fig, figures_dir, "checkpoint_comparison", outputs_figures_dir)
    plt.close()

    # -------------------------------------------------------------
    # Figure 6: Corpus Split & Packing Efficiency Summary
    # -------------------------------------------------------------
    corpus_manifest_path = data_dir / "corpus_manifest.json"
    split_manifest_path = data_dir / "split_manifest.json"

    if corpus_manifest_path.exists() and split_manifest_path.exists():
        corpus_data = load_json(corpus_manifest_path)
        split_data = load_json(split_manifest_path)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Split document counts
        counts = split_data.get("document_counts", {})
        split_labels = [f"Train (90%)\n{counts.get('train', 87174):,} docs",
                        f"Val (5%)\n{counts.get('val', 4843):,} docs",
                        f"Test (5%)\n{counts.get('test', 4844):,} docs"]
        split_sizes = [counts.get("train", 87174), counts.get("val", 4843), counts.get("test", 4844)]
        ax1.pie(split_sizes, labels=split_labels, autopct="%1.1f%%", colors=["#2b5c8f", "#d95f02", "#7570b3"], startangle=140, explode=(0.05, 0.08, 0.08), textprops={'fontweight': 'bold'})
        ax1.set_title("Deterministic 90/5/5 Document Partition\n(Total: 96,861 Documents)", fontweight="bold")

        # Packing efficiency comparison
        pack_labels = ["Train Set\n(10,484 seqs)", "Val Set\n(585 seqs)"]
        efficiencies = [99.99, 99.89]
        bars = ax2.bar(pack_labels, efficiencies, color=["#31a354", "#74c476"], edgecolor="black", alpha=0.85, width=0.45)
        ax2.set_title("512-Token Sequence Packing Efficiency\n(Zero Token Splitting / Minimal Waste)", fontweight="bold")
        ax2.set_ylabel("Packing Efficiency (%)")
        ax2.set_ylim(98.0, 100.2)
        for bar in bars:
            h = bar.get_height()
            waste = 100.0 - h
            ax2.annotate(f"{h:.2f}%\n(Waste: {waste:.2f}%)", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

        plt.suptitle("Maritime Corpus Provenance & Token Packing Summary", fontweight="bold", y=1.03)
        save_fig_dual(fig, figures_dir, "corpus_split_packing_summary", outputs_figures_dir)
        plt.close()

    # Also generate individual corpus length distribution and split distribution for backwards compatibility
    if corpus_manifest_path.exists():
        corpus_data = load_json(corpus_manifest_path)
        buckets = corpus_data.get("length_buckets", {})
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = [k.replace("_words", " words").replace("_", "–") for k in buckets.keys()]
        values = list(buckets.values())
        bars = ax.bar(labels, values, color="#2b5c8f", edgecolor="black", alpha=0.85)
        ax.set_title("Maritime Corpus Document Length Distribution (96,861 Docs)", pad=15, fontweight="bold")
        ax.set_xlabel("Word Length Range")
        ax.set_ylabel("Document Count")
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f"{height:,}", xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
        save_fig_dual(fig, figures_dir, "corpus_length_distribution", outputs_figures_dir)
        plt.close()

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
        ax.set_title("Deterministic Document Partitioning (96,861 Docs)", pad=15, fontweight="bold")
        ax.set_ylabel("Document Count")
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:,}", xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
        save_fig_dual(fig, figures_dir, "split_distribution", outputs_figures_dir)
        plt.close()

    tok_report_path = data_dir / "tokenizer_report.json"
    if tok_report_path.exists():
        tok_data = load_json(tok_report_path)
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
            ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
        save_fig_dual(fig, figures_dir, "tokenizer_fertility", outputs_figures_dir)
        plt.close()

    print("All figures successfully generated in PNG and PDF formats.")

if __name__ == "__main__":
    main()
