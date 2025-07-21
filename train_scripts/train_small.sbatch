#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out/train_small.out
#SBATCH -e slurm_out/train_small.err
#SBATCH --container-image="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/container/final_final.sqsh"
#SBATCH --container-mounts="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/data:/mnt/data,/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/models:/mnt/models,/dss/dsshome1/01/ge43teg2/.config/wandb:/root/.config/wandb,/dss/dsshome1/01/ge43teg2/MaPra2025:/workspace"
#SBATCH --chdir=/workspace
#SBATCH --export=NONE
#SBATCH --time=14:00:00
#SBATCH --mem=150G


source ~/.bashrc
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /opt/envs/f_token

export WANDB_API_KEY=$(
  grep -E '^\s*api_key\s*=' /root/.config/wandb/credentials \
    | sed 's/.*=\s*//'
)

export PYTHONPATH=/workspace:$PYTHONPATH

echo "Starting training script..."
python -u models/train.py \
  --model final \
  --lora_plm \
  --lora_decoder \
  --hidden 512 256 256 \
  --kernel 16 3 3 \
  --data_dir /mnt/data/large/subset2/ \
  --epochs 20 \
  --out_folder /mnt/models \
  --batch 8 \
  --patience 20

echo "Training script finished."
