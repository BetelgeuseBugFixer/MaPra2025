#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out/train_c_alpha_std_output.out
#SBATCH -e slurm_out/train_c_alpha_std_err.err
#SBATCH --container-image="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/container/final_final.sqsh"
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/data:/mnt/data,/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/models:/mnt/models
#SBATCH --time=40:00:00
#SBATCH --mem=70G

# command to execute and track it
# sbatch train_scripts/train_c_alpha.sbatch &  tail -f slurm_out/train_c_alpha_std_output.out slurm_out/train_c_alpha_std_err.err

source ~/.bashrc
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /opt/envs/f_token
cd ~/MaPra2025


# Default-Werte für Batch-Size und Learning-Rate
BATCH_SIZE=8
LR=0.0001
LORA_R=10

# Flag-Parsing: -b für Batch-Size, -l für Learning-Rate
while getopts "b:l:z:" opt; do
  case "$opt" in
    b) BATCH_SIZE=$OPTARG ;;
    l) LR=$OPTARG ;;
    z) LORA_R=$OPTARG ;;
    *) echo "Usage: sbatch c_alpha_small.sbatch [-b batch_size] [-l learning_rate] [-z lora_rank]" >&2
       exit 1 ;;
  esac
done

# WandB API Key holen
export WANDB_API_KEY=$(
  grep -E '^\s*api_key\s*=' /mnt/dir/MaPra2025/wandb_key \
    | sed 's/.*=\s*//'
)

export PYTHONPATH=~/MaPra2025:$PYTHONPATH
echo "Starting training script..."
python -u models/train.py --model final_final \
                          --c_alpha \
                          --lora_plm \
                          --lora_decoder \
                          --hidden 1024 512 256 \
                          --kernel 17 5 3 \
                          --data_dir /mnt/data/large/subset2/ \
                          --epochs 30 \
                          --out_folder /mnt/models \
                          --batch $BATCH_SIZE \
                          --patience 20 \
                          --lr $LR \
                          --lora_r $LORA_R \

echo "Training script finished."
