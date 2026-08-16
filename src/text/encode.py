"""Chunked BERT document encoding with mean pooling and disk caching."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "interim" / "encode_cache"


def _cache_key(texts: list[str], model_name: str, chunk_tokens: int) -> str:
    hasher = hashlib.sha256()
    hasher.update(model_name.encode())
    hasher.update(str(chunk_tokens).encode())
    hasher.update(str(len(texts)).encode())
    for text in texts:
        hasher.update(text.encode("utf-8", errors="ignore"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def encode_documents(
    texts: list[str],
    model_name: str = "bert-base-uncased",
    chunk_tokens: int = 510,
    batch_size: int = 16,
) -> np.ndarray:
    cache_path = CACHE_DIR / f"{_cache_key(texts, model_name, chunk_tokens)}.npy"
    if cache_path.exists():
        print(f"encode_documents: loaded {len(texts)} docs from cache ({cache_path.name})")
        return np.load(cache_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"encode_documents: running on {device}")

    # Flatten every document into its chunks up front, tracking which
    # document each chunk belongs to, so chunks can be batched across
    # document boundaries for throughput rather than one document at a time.
    chunk_input_ids: list[list[int]] = []
    doc_of_chunk: list[int] = []
    for doc_idx, text in enumerate(texts):
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=chunk_tokens,
            return_overflowing_tokens=True,
            padding=False,
        )
        for ids in encoded["input_ids"]:
            chunk_input_ids.append(ids)
            doc_of_chunk.append(doc_idx)
    doc_of_chunk = np.array(doc_of_chunk)

    hidden_dim = model.config.hidden_size
    chunk_vectors = np.zeros((len(chunk_input_ids), hidden_dim), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(chunk_input_ids), batch_size):
            batch_ids = chunk_input_ids[start : start + batch_size]
            batch = tokenizer.pad({"input_ids": batch_ids}, padding=True, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
            mask = batch["attention_mask"].unsqueeze(-1)  # (batch, seq_len, 1)

            # Mean-pool over non-padding tokens only.
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = (summed / counts).cpu().numpy()

            chunk_vectors[start : start + len(batch_ids)] = pooled

    doc_vectors = np.zeros((len(texts), hidden_dim), dtype=np.float32)
    for doc_idx in range(len(texts)):
        chunks_for_doc = chunk_vectors[doc_of_chunk == doc_idx]
        if len(chunks_for_doc) > 0:
            doc_vectors[doc_idx] = chunks_for_doc.mean(axis=0)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, doc_vectors)
    return doc_vectors
