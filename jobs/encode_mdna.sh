#!/bin/bash
#$ -cwd
#$ -N encode_mdna
#$ -j y
#$ -o logs/
#$ -t 1-20
#$ -q all.q
#$ -l m_mem_free=8G

# Must match the cache populated by src/text/download_model.py on the
# login node beforehand — compute nodes have no network access, so
# AutoModel.from_pretrained() must find the weights here, not fetch them.
export HF_HOME="$HOME/accy575/data/interim/hf_cache"
# Forces cache-only lookups: fails fast and clearly if a file is missing,
# instead of hanging on a network call that can never succeed here.
export HF_HUB_OFFLINE=1
# Without this, PyTorch's OpenMP backend tries to spawn one thread per
# CPU core for a single forward pass, which hit "Resource temporarily
# unavailable" against a tight process-count ulimit in testing. Twenty
# array tasks already provide the real parallelism here, so one thread
# per process is correct anyway, not just a workaround.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHON="$HOME/accy575/.venv/bin/python"
SHARD=$((SGE_TASK_ID - 1))

"$PYTHON" -m src.text.encode_mdna_shard \
    --shard "$SHARD" \
    --nshards 20 \
    --out "data/interim/mdna_embeddings_shard_${SHARD}.npz"
