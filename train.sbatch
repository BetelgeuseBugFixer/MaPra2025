#!/bin/bash
#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out/train_std_output.out
#SBATCH -e slurm_out/train_std_err.err
#SBATCH --container-image="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/container/mapra_container.sqsh"
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/data:/mnt/data,/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/repro/:/mnt/repro

cd /mnt/repro/MaPra2025
export PYTHONPATH=/mnt/repro/MaPra2025:$PYTHONPATH
echo "Starting training script..."
python models/train.py --hidden 10000 8000 4000 --kernel 17 --emb_file /mnt/data/lys6/emb/ly6.h5 --tok_jsonl /mnt/data/lys6/tokens_sanitized.jsonl --epochs 200 --split_file /mnt/data/lys6/split_ids.json --out_folder train_run --batch 1000 --patience 200
echo "Training script finished."
