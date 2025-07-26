#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out/model_preds.out
#SBATCH -e slurm_out/model_preds.out
#SBATCH --container-image="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/container/final_final.sqsh"
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/data:/mnt/data,/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/models:/mnt/models,/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge43teg2/:/mnt/dir
#SBATCH --time=14:00:00
#SBATCH --mem=150G

# command to execute and track it
# sbatch benchmark/model_preds.sbatch & tail -f slurm_out/model_preds.out
source ~/.bashrc
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /opt/envs/f_token

echo "Now in: $(pwd)"

cd /mnt/dir/MaPra2025
export PYTHONPATH=/mnt/dir/MaPra2025:$PYTHONPATH

echo "Now in: $(pwd)"

echo "Starting eval script..."
python -u benchmark/model_predictions.py \
--final /mnt/dir/models/final_k17_5_1_h512_256_128_a_1_b_1_plm_lora_lr0.0001_3/final_k17_5_1_h512_256_128_a_1_b_1_plm_lora_latest.pt \
--bio2token mnt/models/t_fold_bio2token_k17_5_1_h512_256_128_lora_lr0.0001_2/t_fold_bio2token_k17_5_1_h512_256_128_lora_latest.pt \
--foldtoken mnt/models/t_fold_foldtoken_k17_5_1_h512_256_128_lora_lr0.0001/t_fold_foldtoken_k17_5_1_h512_256_128_lora_latest.pt

echo "Eval script finished."
