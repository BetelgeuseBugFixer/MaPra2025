#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out/model_preds_esm.out
#SBATCH -e slurm_out/model_preds_esm.err
#SBATCH --container-image="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/container/final_final.sqsh"
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/data:/mnt/data,/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/models:/mnt/models
#SBATCH --time=2:00:00

# sbatch benchmark/model_preds_esm.sbatch & tail -f slurm_out/model_preds_esm.out slurm_out/model_preds_esm.err
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /opt/envs/f_token
cd ~/MaPra2025
export PYTHONPATH=~/MaPra2025:$PYTHONPATH
echo "Starting eval script..."
python -u benchmark/model_predictions.py \
          --prostt5 /mnt/models/final_final_k17_3_3_h4096_2048_1024_plm_lora_lr5e-05/final_final_k17_3_3_h4096_2048_1024_plm_lora.pt \
          --esm

echo "Eval script finished."