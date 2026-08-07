import tempfile
import shutil
from pathlib import Path
from dapt.src.dataset import DatasetSplitter

def test_dataset_splitter():
    docs = [f"Maritime document text record number {i}" for i in range(100)]
    splitter = DatasetSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    
    train1, val1, test1 = splitter.split(docs)
    assert len(train1) == 80
    assert len(val1) == 10
    assert len(test1) == 10

    # Test seed reproducibility
    train2, val2, test2 = splitter.split(docs)
    assert train1 == train2
    assert val1 == val2
    assert test1 == test2

    temp_dir = tempfile.mkdtemp()
    try:
        manifest = splitter.prepare_and_save(docs, temp_dir)
        assert manifest["document_counts"]["train"] == 80
        assert Path(manifest["file_paths"]["train"]).exists()
        assert Path(manifest["file_paths"]["val"]).exists()
        assert Path(manifest["file_paths"]["test"]).exists()
    finally:
        shutil.rmtree(temp_dir)
