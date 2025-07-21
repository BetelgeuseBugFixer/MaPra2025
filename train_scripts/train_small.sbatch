#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out/train_small.out
#SBATCH -e slurm_out/train_small.err
#SBATCH --container-image="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/container/final_final.sqsh"
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/data:/mnt/data,/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge43teg2/:/mnt/dir
#SBATCH --time=7:00:00
#SBATCH --mem=150G

source ~/.bashrc
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /opt/envs/f_token

echo "Now in: $(pwd)"
cd /mnt/dir/MaPra2025
echo "Now in: $(pwd)"
ls -lh

# WandB API Key
export WANDB_API_KEY=$(
  grep -E '^\s*api_key\s*=' /mnt/dir/MaPra2025/wandb_key \
    | sed 's/.*=\s*//'
)

echo "Now in: $(pwd)"

echo "Starting training script..."
python -u models/train.py \
  --model final \
  --lora_decoder \
  --hidden 512 256 256 \
  --kernel 16 3 3 \
  --data_dir /mnt/data/large/subset2/ \
  --epochs 20 \
  --out_folder /mnt/dir/models \
  --batch 8 \
  --patience 20 \
  --lr 0.001

echo "Training script finished."
