#!/bin/bash
#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out/train_%j.out
#SBATCH -e slurm_out/train_%j.err
#SBATCH --container-image="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/container/mapra_container.sqsh"
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/data:/mnt/data,/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/repro/:/mnt/repro

cd /mnt/repro/MaPra2025
export PYTHONPATH=/mnt/repro/MaPra2025:$PYTHONPATH

python models/train.py \
  --hidden=__HIDDEN_SIZE__ \
  --dropout=__DROPOUT__ \
  --learning-rate=__LEARNING_RATE__ \
  --batch=__BATCH_SIZE__ \
  --kernel_size=__KERNEL_SIZE__ \
  --out_folder=__OUT_FOLDER__ \
  --emb_file /mnt/data/lys6/emb/ly6.h5 \
  --tok_jsonl /mnt/data/lys6/tokens_sanitized.jsonl \
  --split_file /mnt/data/lys6/split_ids.json \
  --epochs 100
