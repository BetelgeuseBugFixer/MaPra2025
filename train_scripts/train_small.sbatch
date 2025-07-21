#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out/train_small.out
#SBATCH -e slurm_out/train_small.err
#SBATCH --container-image="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/container/final_final.sqsh"
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/data:/mnt/data,/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge43teg2/models:/mnt/models,/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge43teg2/MaPra2025:/workspace
#SBATCH --time=7:00:00
#SBATCH --mem=150G

source ~/.bashrc
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /opt/envs/f_token

export PYTHONPATH=/workspace:$PYTHONPATH

# WandB API Key
export WANDB_API_KEY=$(
  grep -E '^\s*api_key\s*=' /workspace/wandb_key \
    | sed 's/.*=\s*//'
)

KEY="$WANDB_API_KEY"
len=${#KEY}
prefix=${KEY:0:4}
suffix=${KEY:len-4:4}
middle=$(printf '%*s' $((len-8)) '' | tr ' ' '*')

echo "WANDB_API_KEY is ${prefix}${middle}${suffix}"
echo "WANDB_ENTITY=${WANDB_ENTITY:-<not set>}"
echo "WANDB_PROJECT=${WANDB_PROJECT:-<not set>}"

cd mnt/workspace

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
