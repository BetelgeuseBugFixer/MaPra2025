#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out/train_tfold_std_output.out
#SBATCH -e slurm_out/train_tfold_std_err.err
#SBATCH --container-image="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/container/final_final.sqsh"
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/data:/mnt/data,/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/repro/:/mnt/repro
#SBATCH --time=01:00:00

# command to execute and track it
# sbatch train_tfold.sbatch &  tail -f slurm_out/train_tfold_std_output.out slurm_out/train_tfold_std_err.err
source ~/.bashrc
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /opt/envs/f_token
cd /mnt/repro/MaPra2025
export PYTHONPATH=/mnt/repro/MaPra2025:$PYTHONPATH
echo "Starting training script..."
python models/train.py --model t_fold --hidden  2048 2048 --kernel 5 5 --data_dir /mnt/data/large/subset/ --epochs 5 --out_folder train_run --batch 16 --patience 10 --lora_plm
echo "Training script finished."
