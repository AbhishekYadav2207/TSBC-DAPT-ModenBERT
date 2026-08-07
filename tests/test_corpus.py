import tempfile
from pathlib import Path
import pytest
from dapt.src.corpus import CorpusLoader

def test_corpus_loader_and_sha256():
    sample_text = "Doc line 1\nDoc line 1 continuation\n\nDoc line 2\n\nDoc line 3"
    with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as f:
        f.write(sample_text)
        temp_path = f.name

    try:
        loader = CorpusLoader(temp_path)
        sha = loader.compute_sha256()
        assert len(sha) == 64
        
        docs = loader.load_documents()
        assert len(docs) == 3
        assert docs[0] == "Doc line 1 Doc line 1 continuation"
        assert docs[1] == "Doc line 2"
        assert docs[2] == "Doc line 3"

        manifest = loader.analyze_corpus()
        assert manifest["total_documents"] == 3
        assert manifest["sha256"] == sha
        assert "quality_caveats" in manifest
    finally:
        Path(temp_path).unlink(missing_ok=True)
