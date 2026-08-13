"""One-off: pre-download the BERT model into the shared cache on scratch.

Run this manually on the WRDS login node (has internet access) BEFORE
submitting jobs/encode_mdna.sh. Compute nodes have no network, so the
model must already be cached locally under $HF_HOME before the array
job starts, or every task will fail trying to reach huggingface.co.

Uses huggingface_hub.snapshot_download rather than
AutoModel/AutoTokenizer.from_pretrained deliberately: the login node caps
per-process virtual memory at 4GB (a hard limit, not raisable), and just
importing torch + transformers pushes close to that ceiling on its own.
snapshot_download only needs huggingface_hub — no torch import at all —
so it fits comfortably. Actually loading the model happens later, only on
compute nodes, which aren't under this constraint.

    ssh wrds
    cd ~/accy575
    export HF_HOME=~/accy575/data/interim/hf_cache
    .venv/bin/python -m src.text.download_model
"""

from huggingface_hub import snapshot_download

MODEL_NAME = "bert-base-uncased"

# bert-base-uncased ships weights for every framework (PyTorch, TF, Flax,
# ONNX, rust) — several GB combined. We only load it via torch/safetensors,
# so restrict to that plus config/tokenizer files.
ALLOW_PATTERNS = ["*.json", "*.txt", "*.safetensors"]

if __name__ == "__main__":
    snapshot_download(repo_id=MODEL_NAME, allow_patterns=ALLOW_PATTERNS)
    print(f"Cached {MODEL_NAME} under $HF_HOME")
