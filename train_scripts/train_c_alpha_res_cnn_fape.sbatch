#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out/train_c_alpha_res_cnn_fape_std_output.out
#SBATCH -e slurm_out/train_c_alpha_res_cnn_fape_std_err.err
#SBATCH --container-image="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/container/final_final.sqsh"
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/data:/mnt/data,/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/models:/mnt/models
#SBATCH --time=40:00:00
#SBATCH --mem=70G

# command to execute and track it
# sbatch train_scripts/train_c_alpha_res_cnn_fape.sbatch &  tail -f slurm_out/train_c_alpha_res_cnn_fape_std_output.out slurm_out/train_c_alpha_res_cnn_fape_std_err.err

source ~/.bashrc
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /opt/envs/f_token
cd ~/MaPra2025
export PYTHONPATH=~/MaPra2025:$PYTHONPATH
echo "Starting training script..."
python -u models/train.py --model final_final \
                          --c_alpha \
                          --res_cnn \
                          --lora_plm \
                          --lora_decoder \
                          --hidden 2048 \
                          --kernel 7 3 3 3 \
                          --data_dir /mnt/data/large/subset2/ \
                          --epochs 30 \
                          --out_folder /mnt/models \
                          --batch 8 \
                          --patience 20 \
                          --lr 0.0003 \
                          --losses fape \
                          --eta_min 0.00005
echo "Training script finished."
