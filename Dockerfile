FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PATH="${HOME}/miniconda3/bin:${PATH}"
ARG PATH="${HOME}/miniconda3/bin:${PATH}"

RUN apt-get update &&  \
    apt-get upgrade -y &&  \
    apt-get install -y \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    gcc \
    locales \
    wget \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    git lfs install

WORKDIR /app
ENV HOME=/app

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /app/miniconda \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
ENV PATH /app/miniconda/bin:$PATH

RUN conda create -p /app/env -y python=3.8

SHELL ["conda", "run","--no-capture-output", "-p","/app/env", "/bin/bash", "-c"]

RUN conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
RUN pip install https://github.com/abhishekkrthakur/xformers/raw/main/xformers-0.0.15%2B7e05e2c.d20221223-cp38-cp38-linux_x86_64.whl

COPY requirements.txt /app/requirements.txt
RUN pip install -U --no-cache-dir -r /app/requirements.txt

COPY . /app
RUN cd /app && pip install -e .
