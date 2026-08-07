import tempfile
import shutil
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM
from dapt.src.checkpointing import CheckpointManager

def test_checkpoint_manager():
    temp_dir = tempfile.mkdtemp()
    try:
        checkpoints_dir = Path(temp_dir) / "checkpoints"
        export_dir = Path(temp_dir) / "export"

        mgr = CheckpointManager(
            checkpoints_dir=str(checkpoints_dir),
            export_dir=str(export_dir),
            save_total_limit=2
        )

        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        model = AutoModelForMaskedLM.from_pretrained("answerdotai/ModernBERT-base")

        # Save step 10
        ckpt10 = mgr.save_checkpoint(step=10, model=model, tokenizer=tokenizer, metrics={"loss": 2.5})
        assert ckpt10.exists()
        assert (ckpt10 / "config.json").exists()

        # Save step 20
        ckpt20 = mgr.save_checkpoint(step=20, model=model, tokenizer=tokenizer, metrics={"loss": 2.3}, is_best=True)
        assert ckpt20.exists()
        assert (checkpoints_dir / "best").exists()

        # Verify latest checkpoint finder
        latest = CheckpointManager.get_latest_checkpoint(str(checkpoints_dir))
        assert latest.name == "checkpoint-20"

        # Export final model
        export_path = mgr.export_final_model(model, tokenizer, {"status": "complete"})
        assert (export_path / "config.json").exists()
        assert (export_path / "experiment_manifest.json").exists()

    finally:
        shutil.rmtree(temp_dir)
