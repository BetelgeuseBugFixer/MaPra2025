#!/bin/bash
#SBATCH -p lrz-v100x2,lrz-hpe-p100x4,lrz-dgx-1-p100x8,rz-dgx-1-v100x8
#SBATCH -o slurm_out/foldtoken_test_std_output.out
#SBATCH -e slurm_out/foldtoken_test_std_err.err
#SBATCH --container-image="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/container/final_final.sqsh"
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/data:/mnt/data
#SBATCH --get-user-env


# sbatch test_foldtoken.sbatch & tail -f slurm_out/foldtoken_test_std_output.out slurm_out/foldtoken_test_std_err.err
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /opt/envs/f_token
cd ~/MaPra2025
export PYTHONPATH=~/MaPra2025:$PYTHONPATH
python -u models/test_foldtoken.py

