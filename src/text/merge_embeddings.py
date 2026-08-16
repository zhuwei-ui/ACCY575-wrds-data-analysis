"""Merge per-shard MD&A embedding files into one array plus a keys table."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

INTERIM_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "interim"
MDNA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "mdna.parquet"
OUTPUT_EMBEDDINGS = INTERIM_DIR / "mdna_embeddings.npy"
OUTPUT_KEYS = INTERIM_DIR / "mdna_embeddings_keys.parquet"
EXPECTED_SHARDS = 20  # must match jobs/encode_mdna.sh's #$ -t 1-20


def main() -> None:
    shard_paths = sorted(INTERIM_DIR.glob("mdna_embeddings_shard_*.npz"))

    if len(shard_paths) != EXPECTED_SHARDS:
        print(
            f"FATAL: found {len(shard_paths)} shard files, expected {EXPECTED_SHARDS}",
            file=sys.stderr,
        )
        sys.exit(1)

    embeddings_parts, gvkey_parts, fyear_parts = [], [], []
    for path in shard_paths:
        # gvkey is a string column, saved as a NumPy object array — needs
        # allow_pickle=True to deserialize (safe here: these are our own
        # freshly-written shard files, not untrusted input).
        data = np.load(path, allow_pickle=True)
        embeddings_parts.append(data["embeddings"])
        gvkey_parts.append(data["gvkey"])
        fyear_parts.append(data["fyear"])

    embeddings = np.concatenate(embeddings_parts, axis=0)
    keys = pd.DataFrame({"gvkey": np.concatenate(gvkey_parts), "fyear": np.concatenate(fyear_parts)})

    # Shards interleave rows (index % nshards), so concatenation order
    # isn't sorted — re-sort to match mdna.parquet's own row order.
    order = keys.sort_values(["gvkey", "fyear"]).index.to_numpy()
    embeddings = embeddings[order]
    keys = keys.iloc[order].reset_index(drop=True)

    n_expected = len(pd.read_parquet(MDNA_PATH))
    if len(embeddings) != n_expected:
        print(
            f"FATAL: merged {len(embeddings)} embeddings, expected {n_expected} (mdna.parquet row count)",
            file=sys.stderr,
        )
        sys.exit(1)

    np.save(OUTPUT_EMBEDDINGS, embeddings)
    keys.to_parquet(OUTPUT_KEYS, index=False)

    print(f"Rows: {len(embeddings)}")
    print(f"Shard files read: {len(shard_paths)}")


if __name__ == "__main__":
    main()
