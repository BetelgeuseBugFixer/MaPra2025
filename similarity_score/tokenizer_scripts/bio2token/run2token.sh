#!/bin/bash
# #SBATCH -p lrz-v100x2 
# #SBATCH --gres=gpu:1
# #SBATCH -o /dss/dsshome1/08/ge43vab2/mapra/data/bio2token_out/output.out
# #SBATCH -e /dss/dsshome1/08/ge43vab2/mapra/data/bio2token_out/std_err.err

# enroot start bio2token
data="/dss/dsshome1/08/ge43vab2/mapra/data/casps/casp14/pdb-domain"
output="/dss/dsshome1/08/ge43vab2/mapra/data/bio2token_out/casp14"

for file in "$data"/*; do
    echo "Processing $file"
    python3 change_config.py $file $output
    uv run scripts/test_pdb.py --config test_pdb.yaml
done