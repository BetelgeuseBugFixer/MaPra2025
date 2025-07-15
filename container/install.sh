#enroot import docker://nvcr.io/nvidia/pytorch:24.12-py3
#enroot create --name final 
# -> python container with python 3.12 and cuda 12.6

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# create conda

# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# chmod +x Miniconda3-latest-Linux-x86_64.sh
# ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda
# source /opt/miniconda/etc/profile.d/conda.sh

# conda create -y -p /opt/envs/f_token python=3.12
conda activate /opt/envs/f_token

# install pytorch and dependencies
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

pip install torch_geometric -f https://data.pyg.org/whl/torch-2.6.0+cu126.html 
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

pip install transformers
pip install sentencepiece
pip install lightning
pip install omegaconf
pip install tqdm
pip install h5py
pip install biotite
pip install peft
pip install matplotlib
pip install wandb

pip install https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install python-box
pip install biopython
pip install nglview
pip install dill
pip install invariant-point-attention
pip install kaleido
pip install pytest
pip install seaborn
pip install hydra-zen
pip install loguru
pip install mmtf-python
pip install pandas
pip install einx

