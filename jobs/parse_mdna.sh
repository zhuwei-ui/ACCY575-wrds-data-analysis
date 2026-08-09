#!/bin/bash
#$ -cwd
#$ -N parse_mdna
#$ -j y
#$ -o logs/
#$ -t 1-20
#$ -q all.q

PYTHON="$HOME/accy575/.venv/bin/python"
SHARD=$((SGE_TASK_ID - 1))

"$PYTHON" src/parse_mdna_shard.py \
    --shard "$SHARD" \
    --nshards 20 \
    --manifest data/interim/mdna_manifest.parquet \
    --out "data/interim/mdna_shard_${SHARD}.parquet"
