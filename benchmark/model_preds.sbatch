#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out/model_preds.out
#SBATCH -e slurm_out/model_preds.out
#SBATCH --container-image="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/container/final_final.sqsh"
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/data:/mnt/data,/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/models:/mnt/models,/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ge43teg2/:/mnt/dir
#SBATCH --time=2:00:00
#SBATCH --mem=150G

set -euo pipefail

# sbatch benchmark/model_preds.sbatch & tail -f slurm_out/model_preds.out
source ~/.bashrc
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /opt/envs/f_token

echo "Now in: $(pwd)"

cd /mnt/dir/MaPra2025

# Make sure our repo wins import resolution & ignore user site packages
export PYTHONPATH=/mnt/dir/MaPra2025:${PYTHONPATH:-}
export PYTHONNOUSERSITE=1

echo "Now in: $(pwd)"

# --- one-shot debug: which whole_model is imported? what's the signature? ---
python - <<'PY'
import sys, inspect
import models.end_to_end.whole_model as wm
print("[debug] whole_model file:", wm.__file__)
print("[debug] FinalFinalModel signature:", inspect.signature(wm.FinalFinalModel.__init__))
print("[debug] sys.path head:", sys.path[:5])
PY

echo "Starting eval script..."
python -u benchmark/model_predictions.py \
  --prostt5 \
  /mnt/models/final_final_prott5_cnn_type_1_k7_3_3_3_h1024_plm_lora_lr5e-05_2/final_final_prott5_cnn_type_1_k7_3_3_3_h1024_plm_lora_latest.pt \
  /mnt/models/final_final_prott5_cnn_type_1_k7_3_3_3_h1024_plm_lora_lr5e-05/final_final_prott5_cnn_type_1_k7_3_3_3_h1024_plm_lora_latest.pt
echo "Eval script finished."
