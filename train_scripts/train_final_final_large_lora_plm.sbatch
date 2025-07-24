#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out/train_final_final_lora_std_output.out
#SBATCH -e slurm_out/train_final_final_lora_std_err.err
#SBATCH --container-image="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/container/final_final.sqsh"
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/data:/mnt/data,/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/models:/mnt/models
#SBATCH --time=48:00:00

# command to execute and track it
# sbatch train_scripts/train_final_final_lora_plm.sbatch &  tail -f slurm_out/train_final_final_lora_std_output.out slurm_out/train_final_final_lora_std_err.err
source ~/.bashrc
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /opt/envs/f_token
cd ~/MaPra2025
export PYTHONPATH=~/MaPra2025:$PYTHONPATH
echo "Starting training script..."
python -u models/train.py --model final_final \
                          --lora_plm \
                          --lora_decoder \
                          --hidden 16384 8192 2048 \
                          --kernel 21 3 3 \
                          --data_dir /mnt/data/large/subset2/ \
                          --epochs 32 \
                          --out_folder /mnt/models \
                          --batch 8 \
                          --patience 6 \
                          --lr 0.0001
echo "Training script finished."
