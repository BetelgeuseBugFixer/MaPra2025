#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out/model_preds_esm.out
#SBATCH -e slurm_out/model_preds_esm.err
#SBATCH --container-image="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/container/final_final.sqsh"
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/data:/mnt/data,/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/models:/mnt/models
#SBATCH --time=1:00:00

# sbatch benchmark/model_preds_esm.sbatch & tail -f slurm_out/model_preds_esm.out slurm_out/model_preds_esm.err
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /opt/envs/f_token
cd ~/MaPra2025
export PYTHONPATH=~/MaPra2025:$PYTHONPATH
echo "Starting eval script..."
python benchmark/model_predictions.py \
          --prostt5 \
          /mnt/models/final_final_prott5_res_cnn_k7_3_3_3_h2048_ca_plm_lora_8_lr8e-05_fape_lddt/final_final_prott5_res_cnn_k7_3_3_3_h2048_ca_plm_lora_8.pt \
          /mnt/models/final_final_prott5_res_cnn_k7_3_3_3_h2048_ca_plm_lora_8_lr0.0003_fape_lddt/final_final_prott5_res_cnn_k7_3_3_3_h2048_ca_plm_lora_8.pt

echo "Eval script finished."