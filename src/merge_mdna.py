"""Merge per-shard MD&A parquet files into a single validated dataset."""

import sys
from pathlib import Path

import pandas as pd

from src.schema import MdnaSchema

INTERIM_DIR = Path(__file__).resolve().parent.parent / "data" / "interim"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "mdna.parquet"
EXPECTED_SHARDS = 20  # must match jobs/parse_mdna.sh's #$ -t 1-20


def main() -> None:
    shard_paths = sorted(INTERIM_DIR.glob("mdna_shard_*.parquet"))

    if len(shard_paths) != EXPECTED_SHARDS:
        print(
            f"FATAL: found {len(shard_paths)} shard files, expected {EXPECTED_SHARDS}",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.concat((pd.read_parquet(p) for p in shard_paths), ignore_index=True)

    dupes = df.duplicated(subset=["gvkey", "fyear"], keep=False)
    assert not dupes.any(), f"{dupes.sum()} duplicate (gvkey, fyear) rows across shards"

    df = df.sort_values(["gvkey", "fyear"]).reset_index(drop=True)
    df = MdnaSchema.validate(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Rows: {len(df)}")
    print(f"Median char_count: {df['char_count'].median()}")
    print(f"Shard files read: {len(shard_paths)}")


if __name__ == "__main__":
    main()
