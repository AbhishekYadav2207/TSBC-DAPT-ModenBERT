from transformers import AutoTokenizer
from dapt.src.packing import DocumentPacker

def test_document_packer():
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    docs = [
        "Vessel CARL EMMA experienced minor engine trouble.",
        "Clear weather and calm 0 meter seas reported.",
        "Crewman rescued after minor electrical fire."
    ]

    packer = DocumentPacker(tokenizer=tokenizer, max_seq_length=128)
    packed_samples, stats = packer.pack_documents(docs)

    assert len(packed_samples) > 0
    assert len(packed_samples[0]["input_ids"]) == 128
    assert len(packed_samples[0]["attention_mask"]) == 128
    assert stats["packing_efficiency_percent"] > 0.0
