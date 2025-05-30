#!/bin/bash

levels=(6 8 10 12)

for level in "${levels[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python foldtoken/reconstruct.py --path_in ~/mapra/data/casps/casp14/pdb-domain/ --path_out ~/mapra/data/foldtoken_out/casp14_out --level $level
    CUDA_VISIBLE_DEVICES=0 python foldtoken/reconstruct.py --path_in ~/mapra/data/casps/casp15/ --path_out ~/mapra/data/foldtoken_out/casp15_out --level $level
done