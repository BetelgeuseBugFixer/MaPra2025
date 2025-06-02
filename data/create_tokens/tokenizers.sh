#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python extract_vq_ids.py --path_in $1 --save_vqid_path $2 --level 8

