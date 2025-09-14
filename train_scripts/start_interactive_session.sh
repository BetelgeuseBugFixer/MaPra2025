salloc -p lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8 --gres=gpu:1
#srun --pty --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/data:/mnt/data,/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/models:/mnt/models \
#     --container-image='/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/container/final_final.sqsh' \
#      bash

srun --pty --container-mounts=/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/data:/mnt/data,/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/models:/mnt/models --container-image='/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/container/final_final.sqsh' bash


#python models/train.py --c_alpha --lora_plm --test_run --hidden 16384 8192 2048 --kernel 17 3 3 --data_dir /mnt/data/large/subset2 --epochs 50 --out_folder /mnt/models/ --batch 8 --lr 0.0001 --losses rmsd --model final_final