from transformers import AutoTokenizer
from dapt.src.tokenizer import resolve_boundary_token, TokenizerAnalyzer

def test_tokenizer_boundary_resolution():
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    boundary_id, boundary_desc = resolve_boundary_token(tokenizer)
    
    assert isinstance(boundary_id, int)
    assert len(boundary_desc) > 0

def test_tokenizer_analyzer():
    docs = ["The fishing vessel encountered heavy seas in the harbor.", "Tug boat towed cargo ship."]
    analyzer = TokenizerAnalyzer("answerdotai/ModernBERT-base")
    report = analyzer.analyze(docs)

    assert report["vocab_size"] > 0
    assert report["sample_documents"] == 2
    assert "token_length_statistics" in report
