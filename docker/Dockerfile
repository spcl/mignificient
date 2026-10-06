# Multi-stage runtime image for the container executor.
#   builder: CUDA 11.6 toolchain + pinned iceoryx (docker/Dockerfile.iceoryx), compiles MIGnificient (Release, Python 3.9).
#   final:   plain Ubuntu + only the runtime pieces (no toolkit, no cuDNN, no PyTorch; packages bring their own).
# Build: docker build -f docker/Dockerfile --build-arg ICEORYX_IMAGE=spcleth/mignificient:iceoryx-local -t spcleth/mignificient:executor-local .
ARG CUDA_IMAGE=spcleth/mignificient:cuda
ARG ICEORYX_IMAGE=spcleth/mignificient:iceoryx
ARG RUNTIME_BASE=ubuntu:22.04
FROM ${ICEORYX_IMAGE} AS iceoryx

FROM ${CUDA_IMAGE} AS builder
COPY --from=iceoryx /opt/iceoryx /opt/iceoryx
COPY --from=iceoryx /opt/iceoryx2 /opt/iceoryx2

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y libacl1 libacl1-dev python3.9 python3.9-venv python3.9-dev && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2 && \
    update-alternatives --set python3 /usr/bin/python3.9 &&\
    pip3 install pybind11 && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/cuda-11.6/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/cuda-11.6/lib64:/opt/cudnn-8.9.7/lib:/opt/iceoryx/lib:/opt/iceoryx2/lib"

COPY . /opt/mignificient/src
RUN cd /opt/mignificient/src \
    && pybind11_DIR=/usr/local/lib/python3.9/dist-packages/pybind11 cmake -B /opt/mignificient/build -S . \
      -DCMAKE_BUILD_TYPE=Release \
      -DPython_EXECUTABLE=/usr/bin/python3.9 \
      -DBUILD_ICEORYX2=ON \
      -DCUDNN_DIR=/opt/cudnn-8.9.7 \
      -DCMAKE_CUDA_ARCHITECTURES=80 \
      -DCMAKE_PREFIX_PATH="/opt/iceoryx;/opt/iceoryx2" \
    && cmake --build /opt/mignificient/build -j$(nproc) --target executor_cpp gpuless _mignificient

# CUDA stub for libcuda.so.1 (we do not mount a GPU into the container).
RUN ln -s /opt/cuda-11.6/lib64/stubs/libcuda.so /opt/cuda-11.6/lib64/stubs/libcuda.so.1

FROM ${RUNTIME_BASE}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends python3.9 libjsoncpp25 libstdc++6 && \
    apt-get purge -y software-properties-common && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

ARG B=/opt/mignificient/build
COPY --from=builder ${B}/executor/bin ${B}/executor/bin
COPY --from=builder ${B}/executor/libexecutor_lib.so ${B}/executor/libexecutor_lib.so
COPY --from=builder ${B}/executor/mignificient ${B}/executor/mignificient
COPY --from=builder ${B}/gpuless/libgpuless.so ${B}/gpuless/libgpuless.so
# CUDA runtime for libgpuless; the stub libcuda.so.1 stands in for the driver (no GPU in the container).
COPY --from=builder /opt/cuda-11.6/lib64/libcudart.so.11.0* /opt/cuda-11.6/lib64/
COPY --from=builder /opt/cuda-11.6/lib64/stubs/libcuda.so.1 /opt/cuda-11.6/lib64/stubs/libcuda.so.1

# libgpuless is linked against libcudnn.so.8; ship only the 142 KB dispatch library so the preload resolves.
# The cuDNN implementation libraries are not shipped (packages bring their own).
COPY --from=builder /opt/cudnn-8.9.7/lib/libcudnn.so.8* /opt/cuda-11.6/lib64/

# No /opt/cudnn: function packages bring their own.
ENV LD_LIBRARY_PATH="/opt/cuda-11.6/lib64:/opt/cuda-11.6/lib64/stubs"
WORKDIR /code
