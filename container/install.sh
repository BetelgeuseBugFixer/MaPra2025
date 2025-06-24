# enroot create --name help nvidia+pytorch+22.12-py3.sqsh
# enroot start help
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# chmod +x Miniconda3-latest-Linux-x86_64.sh
# ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda
# source /opt/miniconda/etc/profile.d/conda.sh

# conda create -y -p /opt/envs/f_token python=3.11
# conda activate /opt/envs/f_token

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


python -m pip install --upgrade pip setuptools wheel packaging ninja

#pip install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia


#pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.3.0+cu118.html

# wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.2cxx11abiTRUE-cp39-cp39-linux_x86_64.whl
# pip install flash_attn-2.6.3+cu118torch2.2cxx11abiTRUE-cp39-cp39-linux_x86_64.whl
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu118torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

pip install lightning
pip install omegaconf
pip install tqdm
