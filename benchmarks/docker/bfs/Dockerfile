# nvidia/cuda:11.6.1-devel-ubuntu20.04 amd64
FROM nvidia/cuda@sha256:8d815140703f4960bb1eb68483c4c9d9551173f6e90cbe39f281134b02866725

WORKDIR /app

COPY . /app

RUN apt-get update && \
    apt-get install -y make g++

# Get the compute capability using nvidia-smi
# RUN compute_cap=$(nvidia-smi --query-gpu=compute_capability --format=csv,noheader)

RUN CUDA_ARCH=compute_80

RUN make clean

# make CUDA_ARCH=sm_80 CUDA_DIR=$CUDA_HOME -j8
RUN make CUDA_ARCH=$CUDA_ARCH CUDA_DIR=$CUDA_HOME -j8

ENV INPUTS="/bench/graph1MW_6.txt"

CMD ./bfs ${INPUTS}