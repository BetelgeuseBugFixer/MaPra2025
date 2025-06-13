#!/bin/bash
#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH -o train_std_output.out
#SBATCH -e train_std_err.err

# create container if it does not exist yet
CONTAINER=mapra_container

if enroot list | grep -q $CONTAINER; then
    echo "Container existiert"
else
    echo "Container existiert nicht"
fi