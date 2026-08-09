#!/bin/bash
#$ -cwd
#$ -N merge_mdna
#$ -j y
#$ -o logs/
#$ -q all.q
#$ -l m_mem_free=8G

PYTHON="$HOME/accy575/.venv/bin/python"

"$PYTHON" -m src.merge_mdna
