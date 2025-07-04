#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out/foldtoken_test_std_output.out
#SBATCH -e slurm_out/foldtoken_test_std_err.err
#SBATCH --container-image="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/container/final.sqsh"
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/data:/mnt/data,/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/repro/:/mnt/repro
#SBATCH --get-user-env


source /opt/miniconda/etc/profile.d/conda.sh
conda activate /opt/envs/f_token

cd /mnt/repro/MaPra2025
python models/test_foldtoken.py

