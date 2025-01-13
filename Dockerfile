FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

ARG USER_NAME

RUN apt-get update && apt-get install -y \
    libopenexr-dev \
    g++ \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists

RUN pip install \
    numpy \
    scipy \
    sets \
    future \
    scikit-image \
    ninja \
    h5py==3.10.0 \
    prefetch-generator==1.0.3 \
    matplotlib \
    opencv-python \
    wandb \
    pyexr \
    einops \
    natten==0.17.1+torch220cu121 -f https://shi-labs.com/natten/wheels/

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    mesa-utils \
    libglib2.0-0


# 변경된 VOLUME 경로
VOLUME /workspace/Data
VOLUME /workspace/Code
VOLUME /workspace/Result

# 변경된 WORKDIR 경로
WORKDIR /workspace/Code