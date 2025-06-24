FROM nvcr.io/nvidia/pytorch:22.05-py3
LABEL maintainer="aaron"

# System Tools & Build Essentials
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/Europe/Berlin /etc/localtime && \
    apt-get update && apt-get install -y \
    build-essential git curl wget ninja-build \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncursesw5-dev libgdbm-dev liblzma-dev tk-dev xz-utils \
    tzdata && \
    dpkg-reconfigure -f noninteractive tzdata

# Python 3.9.17
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/3.9.17/Python-3.9.17.tgz && \
    tar -xzf Python-3.9.17.tgz && \
    cd Python-3.9.17 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    ln -s /usr/local/bin/python3.9 /usr/bin/python && \
    ln -s /usr/local/bin/pip3.9 /usr/bin/pip

RUN /usr/local/bin/pip3.9 install -U pip setuptools wheel packaging


# PyTorch 2.0.1 + cu117
RUN /usr/local/bin/pip3.9 install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 \
        --index-url https://download.pytorch.org/whl/cu117

# Numerik
RUN /usr/local/bin/pip3.9 install "scipy>=1.11,<1.12"

# PyTorch Geometric CUDA wheels
ENV PYG_INDEX=https://data.pyg.org/whl/torch-2.0.1+cu117.html

RUN /usr/local/bin/pip3.9 install --no-index --find-links "$PYG_INDEX" torch-scatter==2.1.2+pt20cu117 && \
    /usr/local/bin/pip3.9 install --no-index --find-links "$PYG_INDEX" torch-sparse==0.6.17+pt20cu117 && \
    /usr/local/bin/pip3.9 install --no-index --find-links "$PYG_INDEX" torch-cluster==1.6.3+pt20cu117 && \
    /usr/local/bin/pip3.9 install --no-index --find-links "$PYG_INDEX" torch-spline-conv==1.2.2+pt20cu117 && \
    /usr/local/bin/pip3.9 install --no-index --find-links "$PYG_INDEX" pyg-lib==0.2.0+pt20cu117 && \
    /usr/local/bin/pip3.9 install --find-links "$PYG_INDEX" torch-geometric==2.3.1

# RUN /usr/local/bin/pip3.9 install flash-attn --no-build-isolation
RUN /usr/local/bin/pip3.9 install pytorch-lightning==1.9.0 omegaconf==2.3.0 tqdm

RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/*


