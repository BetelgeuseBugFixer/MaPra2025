#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out/%j.out
#SBATCH -e slurm_out/%j.err
#SBATCH --container-image="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/container/final_final.sqsh"
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/data:/mnt/data,/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge43teg2/:/mnt/dir
#SBATCH --time=20:00:00
#SBATCH --mem=200G

# Default-Werte für Batch-Size und Learning-Rate
BATCH_SIZE=8
LR=0.0001
BETA=0

# Flag-Parsing: -b für Batch-Size, -l für Learning-Rate
while getopts "b:l:z:" opt; do
  case "$opt" in
    b) BATCH_SIZE=$OPTARG ;;
    l) LR=$OPTARG        ;;
    z) BETA=$OPTARG        ;;
    *) echo "Usage: sbatch train_small.sbatch [-b batch_size] [-l learning_rate] [-z beta]" >&2
       exit 1 ;;
  esac
done

source ~/.bashrc
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /opt/envs/f_token

echo "Now in: $(pwd)"
cd /mnt/dir/MaPra2025
echo "Now in: $(pwd)"

# WandB API Key holen
export WANDB_API_KEY=$(
  grep -E '^\s*api_key\s*=' /mnt/dir/MaPra2025/wandb_key \
    | sed 's/.*=\s*//'
)

# PYTHONPATH auf Projekt-Root setzen
export PYTHONPATH="/mnt/dir/MaPra2025:$PYTHONPATH"

echo "Starting training script with batch_size=$BATCH_SIZE, lr=$LR, beta=$BETA"
python -u models/train.py \
  --model final \
  --lora_plm \
  --lora_decoder \
  --hidden 512 256 128 \
  --kernel 17 5 1 \
  --data_dir /mnt/data/large/subset2/ \
  --epochs 20 \
  --out_folder /mnt/dir/models \
  --batch $BATCH_SIZE \
  --patience 20 \
  --lr $LR \
  --beta $BETA \
  --resume /mnt/dir/models/final_k17_5_1_h512_256_128_a_1_b_0_plm_lora_lr0.0001_2/final_k17_5_1_h512_256_128_a_1_b_0_plm_lora_latest.pt \
  --wandb_resume_id aiszu465

echo "Training script finished."
