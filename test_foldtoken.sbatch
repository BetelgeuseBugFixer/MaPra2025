#!/bin/bash
#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out/foldtoken_test_std_output.out
#SBATCH -e slurm_out/foldtoken_test_std_err.err
#SBATCH --container-image="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/container/help_me.sqsh"
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/data:/mnt/data,/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/repro/:/mnt/repro
#SBATCH --get-user-env

# #export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
# conda init bash
# source ~/.bashrc
# conda activate /opt/envs/f_token
# 1. Init conda properly
source /opt/miniconda/etc/profile.d/conda.sh

# 2. Activate your env by full path
conda activate /opt/envs/f_token

cd /mnt/repro/MaPra2025
python test_foldtoken.py

