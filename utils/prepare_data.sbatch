#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out/prepare_data_std_output.out
#SBATCH -e slurm_out/prepare_data.err
#SBATCH --container-image="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/container/final_final.sqsh"
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/data:/mnt/data
#SBATCH --time=48:00:00

# Activate conda environment
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /opt/envs/f_token

cd ~/MaPra2025
export PYTHONPATH=~/MaPra2025:$PYTHONPATH


python -u utils/prepare_data.py
