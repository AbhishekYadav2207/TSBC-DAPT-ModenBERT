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
        "training_analysis.md", "evaluation_analysis.md", "experiment_summary.md"
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

    # 5. Required Figures in dapt/figures/
    expected_figures = [
        "corpus_length_distribution.png", "split_distribution.png",
        "tokenizer_fertility.png", "validation_loss_curve.png",
        "perplexity_curve.png", "baseline_vs_dapt.png", "checkpoint_comparison.png"
    ]
    for fig in expected_figures:
        fig_path = root / "figures" / fig
        if not fig_path.exists():
            errors.append(f"Missing generated figure: {fig_path}")

    # 6. JSON Artifact Verification
    artifacts = [
        root / "outputs" / "data" / "corpus_manifest.json",
        root / "outputs" / "data" / "split_manifest.json",
        root / "outputs" / "data" / "tokenizer_report.json",
        root / "outputs" / "experiments" / "comparison_report.json",
        root / "outputs" / "experiments" / "baseline-modernbert" / "evaluation_metrics.json"
    ]
    for art in artifacts:
        if not art.exists():
            errors.append(f"Missing required empirical artifact: {art}")

    # 7. Checkpoint Distinction Verification in Docs
    # Check if Step 855 is incorrectly referred to as 'best checkpoint'
    readme_content = ""
    if (root / "README.md").exists():
        with open(root / "README.md", "r", encoding="utf-8") as f:
            readme_content = f.read()
        if "best validation checkpoint (step 855)" in readme_content.lower() or "best checkpoint (step 855)" in readme_content.lower():
            errors.append("Incorrect checkpoint terminology in README.md: Step 855 should be 'released artifact', Step 800 is 'best validation checkpoint'.")

    # 8. Markdown Relative Link Resolution
    all_md_files = list(root.glob("*.md")) + list((root / "docs").glob("*.md")) + list((root / "reports").glob("*.md"))
    link_pattern = re.compile(r'\[([^\]]+)\]\((file:///[^\)]+|[^\)]+)\)')

    for md_file in all_md_files:
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
        for match in link_pattern.finditer(content):
            link_target = match.group(2)
            if link_target.startswith("http://") or link_target.startswith("https://") or link_target.startswith("#"):
                continue
            
            # Handle file:/// links
            clean_target = link_target.replace("file:///", "")
            target_path = Path(clean_target)
            if not target_path.is_absolute():
                target_path = (md_file.parent / clean_target).resolve()
            
            if not target_path.exists():
                warnings.append(f"Broken link in {md_file.relative_to(root)} -> {link_target}")

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
        exit(1)
    else:
        print("\n[SUCCESS] All Documentation & Artifact Validation Checks PASSED Cleanly!")

if __name__ == "__main__":
    main()
