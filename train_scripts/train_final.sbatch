#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out/train_final_std_output.out
#SBATCH -e slurm_out/train_final_std_err.err
#SBATCH --container-image="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/container/final_final.sqsh"
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/data:/mnt/data,/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/models:/mnt/models
#SBATCH --time=9:00:00
#SBATCH --mem=250G

# command to execute and track it
# sbatch train_scripts/train_final.sbatch &  tail -f slurm_out/train_final_std_output.out slurm_out/train_final_std_err.err
source ~/.bashrc
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /opt/envs/f_token
cd ~/MaPra2025
export PYTHONPATH=~/MaPra2025:$PYTHONPATH
echo "Starting training script..."
python -u models/train.py --model final \
                          --lora_decoder \
                          --hidden 1024 \
                          --kernel 3 1 1 \
                          --data_dir /mnt/data/large/subset2/ \
                          --epochs 60 \
                          --out_folder /mnt/models \
                          --batch 8 \
                          --patience 20 \
                          --beta 0 \
                          --lr 0.00001
echo "Training script finished."
