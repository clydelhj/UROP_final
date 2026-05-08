# SynDiff training image for RTX Pro 5000 Blackwell (sm_120) on the school server.
#
# Base: CUDA 12.8.1 + cuDNN, devel variant ships nvcc which the SynDiff custom
# CUDA ops (utils/op/*) need at first run for ninja JIT compilation.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System packages: build toolchain for ninja JIT, plus git/curl for miniforge.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        ninja-build \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Miniforge supplies a clean conda + Python 3.11 env per environment.yml.
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN curl -fsSL -o /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p $CONDA_DIR \
    && rm /tmp/miniforge.sh \
    && conda clean -afy

WORKDIR /app

# Build the conda env first so this layer caches independently of code changes.
# environment.yml pins torch==2.7.1+cu118 and nvidia-*-cu11 wheels, which live
# on PyTorch's index (not PyPI) and are sm_90-max anyway. Strip those lines so
# conda creates a torch-less env, then we install Blackwell-ready cu128 torch
# in the next step (it pulls its own correct nvidia-*-cu12 deps).
COPY environment.yml /app/environment.yml.orig
RUN sed -E '/^[[:space:]]+- (torch==|torchvision==|triton==|nvidia-[a-z-]+-cu11==)/d' \
        /app/environment.yml.orig > /app/environment.yml \
    && conda env create -f /app/environment.yml \
    && conda clean -afy

# Install Blackwell-compatible PyTorch from PyTorch's cu128 index.
# If 2.7.1+cu128 ever stops resolving, bump to 2.8.0 / torchvision 0.23.0.
RUN /opt/conda/envs/syndiff/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.7.1 torchvision==0.22.1

# Make the syndiff env's python the default so `python3` in run_train.sh works.
ENV PATH=/opt/conda/envs/syndiff/bin:$PATH

# Copy the SynDiff source. Dataset and outputs are mounted at runtime, never baked.
COPY . /app

# Mount points and runtime knobs. Override at `docker run` with -e if needed.
ENV INPUT_PATH=/data \
    OUTPUT_PATH=/output \
    CUDA_HOME=/usr/local/cuda \
    CUDA_VISIBLE_DEVICES=0,1,2,3

RUN mkdir -p /data /output

CMD ["bash", "run_train.sh"]
