import os
import re
import json
from pathlib import Path

def load_json(file_path: Path) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    root = Path(__file__).resolve().parent.parent
    errors = []
    warnings = []

    print("==================================================")
    print("      DAPT Documentation & Artifact Validator     ")
    print("==================================================")

    # 1. Required Directory Structure Verification
    required_dirs = [
        root / "docs",
        root / "reports",
        root / "diagrams",
        root / "figures",
        root / "scripts",
        root / "src",
        root / "outputs" / "data",
        root / "outputs" / "experiments"
    ]
    for rdir in required_dirs:
        if not rdir.exists():
            errors.append(f"Missing required directory: {rdir}")

    # 2. Required Markdown Documents in dapt/docs/
    expected_docs = [
        "00_overview.md", "01_architecture.md", "02_corpus_contract.md",
        "03_data_preparation.md", "04_tokenization_packing.md", "05_mlm_training.md",
        "06_checkpointing.md", "07_evaluation.md", "08_results.md",
        "09_reproducibility.md", "10_downstream_usage.md", "11_rag_integration.md",
        "12_future_research.md", "13_api_reference.md"
    ]
    for doc in expected_docs:
        doc_path = root / "docs" / doc
        if not doc_path.exists():
            errors.append(f"Missing documentation page: {doc_path}")

    # 3. Required Reports in dapt/reports/
    expected_reports = [
        "corpus_analysis.md", "dataset_analysis.md", "tokenizer_analysis.md",
        "training_analysis.md", "evaluation_analysis.md", "experiment_summary.md",
        "dapt_report.md", "reproducibility_report.md"
    ]
    for rpt in expected_reports:
        rpt_path = root / "reports" / rpt
        if not rpt_path.exists():
            errors.append(f"Missing analysis report: {rpt_path}")

    # 4. Required Diagrams in dapt/diagrams/
    expected_diagrams = [
        "architecture.md", "data_flow.md", "training_pipeline.md",
        "checkpoint_lifecycle.md", "downstream_tasks.md", "rag_architecture.md"
    ]
    for dia in expected_diagrams:
        dia_path = root / "diagrams" / dia
        if not dia_path.exists():
            errors.append(f"Missing diagram source: {dia_path}")

    # 5. Required Figures in dapt/figures/ (PNG and PDF)
    expected_figures = [
        "validation_loss_curve", "perplexity_curve", "baseline_vs_dapt",
        "learning_rate_trajectory", "checkpoint_comparison",
        "corpus_split_packing_summary", "corpus_length_distribution",
        "split_distribution", "tokenizer_fertility"
    ]
    for fig in expected_figures:
        png_path = root / "figures" / f"{fig}.png"
        pdf_path = root / "figures" / f"{fig}.pdf"
        if not png_path.exists():
            errors.append(f"Missing generated PNG figure: {png_path}")
        if not pdf_path.exists():
            warnings.append(f"Missing vector PDF figure: {pdf_path}")

    # 6. JSON Artifact Verification
    artifacts = [
        root / "outputs" / "data" / "corpus_manifest.json",
        root / "outputs" / "data" / "split_manifest.json",
        root / "outputs" / "data" / "tokenizer_report.json",
        root / "outputs" / "experiments" / "comparison_report.json",
        root / "outputs" / "experiments" / "training_history.json",
        root / "outputs" / "experiments" / "dapt_summary.json",
        root / "outputs" / "experiments" / "dapt_results.json",
        root / "outputs" / "experiments" / "checkpoint_summary.json",
        root / "outputs" / "experiments" / "baseline-modernbert" / "evaluation_metrics.json",
        root / "outputs" / "experiments" / "MaritimeBERT-v1" / "evaluation_metrics.json"
    ]
    for art in artifacts:
        if not art.exists():
            errors.append(f"Missing required empirical artifact: {art}")

    # 7. Checkpoint Distinction Verification in Docs
    # Verify terminology for Step 950 and Step 984 in current DAPT documentation
    readme_path = root / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read().lower()
        if "best validation checkpoint (step 855)" in readme_content or "step 855" in readme_content:
            errors.append("Stale reference to Step 855 found in current dapt/README.md.")
        if "step 800" in readme_content:
            errors.append("Stale reference to Step 800 found in current dapt/README.md.")
        if "best validation checkpoint (step 950)" in readme_content:
            errors.append("Incorrect terminology: Step 950 should be called 'best validation state (unpersisted)', not 'best validation checkpoint'.")

    # 8. Smart Stale-Value Audit for Current DAPT Documentation vs Archived Reports
    stale_patterns = {
        "96,715": "Stale document count 96,715 (current run: 96,861)",
        "3,372,882": "Stale word count 3,372,882 (current run: 3,830,350)",
        "20,671,574": "Stale character count 20,671,574 (current run: 24,356,820)",
        "855 steps": "Stale training steps 855 (current run: 984 steps)",
        "0.5898": "Stale MLM loss 0.5898 from previous run",
        "1.8037": "Stale perplexity 1.8037 from previous run"
    }

    current_dapt_files = list(root.glob("*.md")) + list((root / "docs").glob("*.md")) + list((root / "reports").glob("*.md"))
    for cfile in current_dapt_files:
        # If file is explicitly an archived/historical file, skip or warn
        is_historical = "archive" in cfile.name.lower() or "historical" in cfile.name.lower()
        with open(cfile, "r", encoding="utf-8") as f:
            text = f.read()
        for pat, desc in stale_patterns.items():
            if pat in text:
                if is_historical:
                    warnings.append(f"Historical file {cfile.name} contains historical reference: '{pat}'")
                else:
                    errors.append(f"Current DAPT file {cfile.relative_to(root)} contains stale value '{pat}': {desc}")

    # 9. Multi-layer Metric Consistency Check
    if (root / "outputs" / "experiments" / "MaritimeBERT-v1" / "evaluation_metrics.json").exists():
        eval_m = load_json(root / "outputs" / "experiments" / "MaritimeBERT-v1" / "evaluation_metrics.json")
        comp_m = load_json(root / "outputs" / "experiments" / "comparison_report.json")
        hist_m = load_json(root / "outputs" / "experiments" / "training_history.json")

        if eval_m.get("mlm_loss") != 0.5247:
            errors.append(f"MaritimeBERT-v1 evaluation mlm_loss divergence: expected 0.5247, got {eval_m.get('mlm_loss')}")
        if eval_m.get("perplexity") != 1.69:
            errors.append(f"MaritimeBERT-v1 evaluation perplexity divergence: expected 1.69, got {eval_m.get('perplexity')}")
        if comp_m["metrics"]["dapt_mlm_loss"] != 0.5247:
            errors.append(f"Comparison report dapt_mlm_loss divergence: expected 0.5247, got {comp_m['metrics']['dapt_mlm_loss']}")
        if hist_m["best_validation_state"]["step"] != 950:
            errors.append("training_history.json best_validation_state step is not 950")
        if hist_m["final_training_checkpoint"]["step"] != 984:
            errors.append("training_history.json final_training_checkpoint step is not 984")

    # Print Report
    print(f"Checked Directories: {len(required_dirs)}")
    print(f"Checked Documentation Pages: {len(expected_docs)}")
    print(f"Checked Reports: {len(expected_reports)}")
    print(f"Checked Diagrams: {len(expected_diagrams)}")
    print(f"Checked Figures: {len(expected_figures)}")
    print(f"Checked Artifacts: {len(artifacts)}")

    if warnings:
        print("\n--- WARNINGS ---")
        for w in warnings:
            print(f"[WARN] {w}")

    if errors:
        print("\n--- ERRORS ---")
        for e in errors:
            print(f"[ERROR] {e}")
        print(f"\nValidation FAILED with {len(errors)} error(s).")
        return 1
    else:
        print("\n[SUCCESS] All Documentation, Terminology, and Artifact Validation Checks PASSED Cleanly!")
        return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
