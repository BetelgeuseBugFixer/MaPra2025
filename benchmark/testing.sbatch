#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out/model_preds.out
#SBATCH -e slurm_out/model_preds.out
#SBATCH --container-image="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/container/final_final.sqsh"
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/models:/mnt/models,/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge43teg2/:/mnt/dir
#SBATCH --time=2:00:00
#SBATCH --mem=150G

# command to execute and track it
# sbatch benchmark/model_preds.sbatch & tail -f slurm_out/model_preds.out
source ~/.bashrc
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /opt/envs/f_token


cd /mnt/dir/MaPra2025
export PYTHONPATH=/mnt/dir/MaPra2025:$PYTHONPATH

echo "Starting prediction testing script..."
python -u benchmark/testing.py --model_ckpt mnt/models/final_final_k7_3_3_3_h1024_lr5e-05/final_final_k7_3_3_3_h1024_latest.pt --pdb_dir mnt/dir/MaPra2025/tokenizer_benchmark/casps/casp15_backbone
echo "Prediction testing script finished."

