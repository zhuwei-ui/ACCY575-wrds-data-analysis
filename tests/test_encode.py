from src.text.encode import encode_documents


def test_encode_documents_shape():
    texts = ["Hello world.", "This is a test document.", "Another short piece of text."]
    embeddings = encode_documents(texts)
    assert embeddings.shape == (3, 768)
