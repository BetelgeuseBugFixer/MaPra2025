#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out/train_c_alpha_res_cnn_resume_std_output.out
#SBATCH -e slurm_out/train_c_alpha_res_cnn_resume_std_err.err
#SBATCH --container-image="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/container/final_final.sqsh"
#SBATCH --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/data:/mnt/data,/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/models:/mnt/models
#SBATCH --time=80:00:00
#SBATCH --mem=70G

# command to execute and track it
# sbatch train_scripts/train_c_alpha_res_cnn_resume.sbatch &  tail -f slurm_out/train_c_alpha_res_cnn_resume_std_output.out slurm_out/train_c_alpha_res_cnn_resume_std_err.err

source ~/.bashrc
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /opt/envs/f_token
cd ~/MaPra2025
export PYTHONPATH=~/MaPra2025:$PYTHONPATH
echo "Starting training script..."
python -u models/train.py --model final_final \
                          --resume /mnt/models/final_final_prott5_res_cnn_k7_3_3_3_h2048_ca_plm_lora_8_lr0.0001_lddt \
                          --wandb_resume_id 9crarhs1\
                          --c_alpha \
                          --res_cnn \
                          --lora_plm \
                          --lora_decoder \
                          --hidden 2048 \
                          --kernel 7 3 3 3 \
                          --data_dir /mnt/data/large/subset2/ \
                          --epochs 60 \
                          --out_folder /mnt/models \
                          --batch 8 \
                          --patience 20 \
                          --lr 0.00005 \
                          --eta_min 0.000005

echo "Training script finished."
