FROM spcleth/mignificient:cuda AS cuda
FROM spcleth/mignificient:iceoryx AS iceoryx
FROM spcleth/mignificient:base

# Copy CUDA toolkit
COPY --from=cuda /opt/cuda-11.6 /opt/cuda-11.6
COPY --from=cuda /opt/cudnn-8.9.7 /opt/cudnn-8.9.7

# Copy iceoryx libraries
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

# Set up environment
ENV PATH="/opt/cuda-11.6/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/cuda-11.6/lib64:/opt/cudnn-8.9.7/lib:/opt/iceoryx/lib:/opt/iceoryx2/lib:${LD_LIBRARY_PATH}"

# Copy MIGnificient source
COPY . /opt/mignificient/src

# Build MIGnificient (only executor and gpuless targets)
RUN cd /opt/mignificient/src \
    && pybind11_DIR=/usr/local/lib/python3.9/dist-packages/pybind11 cmake -B /opt/mignificient/build -S . \
      -DCUDNN_DIR=/opt/cudnn-8.9.7 \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_ICEORYX2=ON \
      -DCUDNN_DIR=/opt/cudnn-8.9.7 \
      -DCMAKE_CUDA_ARCHITECTURES=80 \
      -DCMAKE_PREFIX_PATH="/opt/iceoryx;/opt/iceoryx2" \
      #-DCMAKE_CUDA_COMPILER=/opt/cuda-11.6/bin/nvcc \
    && cmake --build /opt/mignificient/build -j$(nproc) --target executor_cpp gpuless _mignificient

# We need this for CUDA stubs to work.
# We do not mount GPU into the container at the runtime.
RUN ln -s /opt/cuda-11.6/lib64/stubs/libcuda.so /opt/cuda-11.6/lib64/stubs/libcuda.so.1

# Runtime environment
ENV LD_LIBRARY_PATH="/opt/cuda-11.6/lib64:/opt/cuda-11.6/lib64/stubs:/opt/cudnn-8.9.7/lib:/opt/iceoryx/lib:/opt/iceoryx2/lib:${LD_LIBRARY_PATH}"

WORKDIR /code
