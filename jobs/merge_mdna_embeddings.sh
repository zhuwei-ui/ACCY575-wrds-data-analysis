#!/bin/bash
#$ -cwd
#$ -N merge_mdna_embeddings
#$ -j y
#$ -o logs/
#$ -q all.q
#$ -l m_mem_free=8G

PYTHON="$HOME/accy575/.venv/bin/python"

"$PYTHON" -m src.text.merge_embeddings
