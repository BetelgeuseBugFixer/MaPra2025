#!/bin/bash

#HIDDEN_SIZES=(768 2048 4096 6144)
#DROPOUTS=(0.2 0.3 0.4)
#LEARNING_RATES=(0.001 0.0001 0.00001)
#BATCH_SIZES=(64 128 512 1024)
#KERNEL_SIZES=(3 5 7 9)

HIDDEN_SIZES=(768 4096)
DROPOUTS=(0.2)
LEARNING_RATES=(0.001)
BATCH_SIZES=(64)
KERNEL_SIZES=(5)

mkdir -p tmp_jobs slurm_out grid_search_out

for hs in "${HIDDEN_SIZES[@]}"; do
  for dr in "${DROPOUTS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
      for bs in "${BATCH_SIZES[@]}"; do
        for ks in "${KERNEL_SIZES[@]}"; do

          job="hs${hs}_dr${dr}_lr${lr}_bs${bs}_ks${ks}"
          jobfile="tmp_jobs/job_${out_dir}.sbatch"
          out_dir="grid_search_out/${job}"
          sed -e "s/__HIDDEN_SIZE__/${hs}/" \
              -e "s/__DROPOUT__/${dr}/" \
              -e "s/__LEARNING_RATE__/${lr}/" \
              -e "s/__BATCH_SIZE__/${bs}/" \
              -e "s/__KERNEL_SIZE__/${ks}/" \
              -e "s/__OUT_DIR__/${out_dir}/" \
              grid_search_templates/template_grid_search.sbatch > "$jobfile"

          sbatch "$jobfile"


        done
      done
    done
  done
done