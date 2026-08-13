"""Encode one shard of MD&A documents into BERT embeddings.

Runs on a compute node inside a Grid Engine array task: no Postgres
connection, no prompts, no network. The model must already be cached
locally (see src/text/download_model.py, which must run on the login
node beforehand — compute nodes can't reach huggingface.co).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.text.encode import encode_documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--nshards", type=int, required=True)
    parser.add_argument("--mdna", type=Path, default=Path("data/raw/mdna.parquet"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    # Sorted so every shard task slices the same deterministic ordering
    # regardless of when/where it runs.
    mdna = pd.read_parquet(args.mdna).sort_values(["gvkey", "fyear"]).reset_index(drop=True)
    shard = mdna[mdna.index % args.nshards == args.shard]

    embeddings = encode_documents(shard["mdna_text"].tolist())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        embeddings=embeddings,
        gvkey=shard["gvkey"].to_numpy(),
        fyear=shard["fyear"].to_numpy(),
    )
    print(f"Wrote {len(shard)} embeddings to {args.out}")


if __name__ == "__main__":
    main()
