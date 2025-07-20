#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out/train_tfold_std_output.out
#SBATCH -e slurm_out/train_tfold_std_err.err
#SBATCH --container-image="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/container/final_final.sqsh"
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/data:/mnt/data,/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/models:/mnt/models
#SBATCH --time=1:00:00

# command to execute and track it
# sbatch train_tfold.sbatch &  tail -f slurm_out/train_tfold_std_output.out slurm_out/train_tfold_std_err.err
source ~/.bashrc
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /opt/envs/f_token
cd ~/MaPra2025
export PYTHONPATH=~/MaPra2025:$PYTHONPATH
echo "Starting training script..."
python -u models/train.py --model t_fold --hidden 10000 8000 4000 --kernel 19 1 1 --data_dir /mnt/data/large/subset2/ --epochs 1 --out_folder /mnt/models --batch 256 --val_batch 64 --patience 10
echo "Training script finished."
