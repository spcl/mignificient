# mignificient



# Dependencies

CUDA 11.6 (required, see below)

cuDNN 8.9.7 for CUDA 11

libacl - if these are not available on your system (usually visible through compilation errors caused by missing headers) and you can't install them as package, then install them locally in `DEPS_PATH`.

https://download.savannah.nongnu.org/releases/acl/acl-2.3.2.tar.xz

pybind11

## Building on this branch (CUDA 11.6 required)

Host builds must use CUDA 11.6, the same version as the executor container image and the conda PyTorch packages.
A server built with another CUDA version reports a different number of device attributes (`CU_DEVICE_ATTRIBUTE_MAX` is 122 in 11.6, 125 in 11.8), which mismatches the 11.6 clients.
CMake does not pick the toolkit for you: without the flags below it uses whatever `nvcc` is first on `PATH` (often `/usr/local/cuda`). Pass the compiler and root explicitly, put 11.6 first on `PATH`, and check `CMakeCache.txt` and `ldd` afterwards.

CUDA 11.6 `nvcc` does not support `sm_89`, so `-DCMAKE_CUDA_ARCHITECTURES=89` fails the compiler check; pass a value it accepts (86 here).
That option only affects the compiler check: `gpuless/CMakeLists.txt` sets `CMAKE_CUDA_ARCHITECTURES "80"` itself and `examples/CMakeLists.txt` sets `80 86`.

```
CUDA=/path/to/cuda-11.6                     # e.g. /opt/cuda/cuda-11.6
# pinned iceoryx2 v0.8.1, installed into build-cu116/iox2-install (needs git, cargo, cmake)
git clone --depth 1 --branch v0.8.1 https://github.com/eclipse-iceoryx/iceoryx2.git build-cu116/iox2-src
cmake -S build-cu116/iox2-src -B build-cu116/iox2-build -DCMAKE_BUILD_TYPE=Release -DBUILD_CXX=ON \
  -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=$PWD/build-cu116/iox2-install
cmake --build build-cu116/iox2-build -j16 && cmake --install build-cu116/iox2-build
export PATH=$CUDA/bin:$PATH
cmake -S . -B build-cu116 -DCMAKE_BUILD_TYPE=Release -DBUILD_ICEORYX2=TRUE -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_COMPILER=$CUDA/bin/nvcc -DCUDAToolkit_ROOT=$CUDA \
  -DCUDNN_DIR=<cudnn-8.9.7-for-cuda11-archive>/ \
  -Dpybind11_DIR=<site-packages>/pybind11/share/cmake/pybind11 \
  -DPython_EXECUTABLE=<python3.9-env>/bin/python \
  -DCMAKE_PREFIX_PATH=$PWD/build-cu116/iox2-install
cmake --build build-cu116 -j16
```

pybind11 is header-only and its CMake config does not depend on the Python version, so `pybind11_DIR` can come from any installed pybind11 (the build used one from a Python 3.10 site-packages).
`-DPython_EXECUTABLE` alone selects the Python for the `_mignificient` executor module; its version must match the interpreter that runs the functions (3.9 for the conda environments used with the container image).
Verify the toolkit:

```
grep -E 'CMAKE_CUDA_COMPILER:|CUDAToolkit_ROOT' build-cu116/CMakeCache.txt
ldd build-cu116/gpuless/manager_device build-cu116/gpuless/libgpuless.so | grep -E 'cuda|cublas'   # only $CUDA/lib64 plus the system libcuda.so.1
```

## Example of building on cluster

We assume that libiberty is installed in `DEPS_PATH`

```
pybind11_DIR=<path-to-your-python/lib/python3.12/site-packages/pybind11 cmake -DCUDNN_DIR=<your-cuddn-installation>/cudnn-8/ -DCMAKE_C_FLAGS="-I ${DEPS_PATH}/include" -DCMAKE_CXX_FLAGS="-I ${DEPS_PATH}/include" -DCMAKE_CXX_STANDARD_LIBRARIES="-L${DEPS_PATH}/lib"  -DCMAKE_BUILD_TYPE=Release ../
```

If JsonCpp is not available, then install it and pass explicitly:

```
jsoncpp_DIR=/path/to/install
```

## Running on the cluster

We assume two environment variables `REPO_DIR` and `BUILD_DIR` that point to source code and build directory, respectively.

### Generate device config

This step only needs to be done once for each node:

```
${REPO_DIR}/tools/list-gpus.sh logs
```

This will create a file `logs/devices.json` with config used later by orchestrator.

### Start Iceoryx's Roudi

This step only needs to be done once when starting MIGnificient on a node. In case of issues with iceoryx, kill `iox-roudi` process and start it again.

```
${REPO_DIR}/tools/start.sh ${BUILD_DIR} logs
```

### Start MIGnificient orchestrator.

There is only one orchestrator per node. Currently, it is recommended to restart orchestrator between experiments (avoids some minor bugs).

The command below starts the orchestrator in the background:

```
${BUILD_DIR}/orchestrator/orchestrator ${BUILD_DIR}/config/orchestrator.json ${REPO_DIR}/logs/devices.json > orchestrator_output.log 2>&1 &
```

In the output file, you should see something similar to:

```
[2025-02-03 20:56:04.324] [info] Reading configuration from /scratch/mcopik/gpus/new_september/build_conda_release/config/orchestrator.json, device database from /scratch/mcopik/gpus/new_september/mignificient/logs/devices.json
2025-02-03 20:56:04.329 [ Debug ]: Application registered management segment 0x15554e240000 with size 65796264 to id 1
2025-02-03 20:56:04.347 [ Debug ]: Application registered payload data segment 0x1553d4e42000 with size 6293584200 to id 2
[2025-02-03 20:56:04.348] [info] Listening on port 10000
```

### Start invoker

This test processes takes a benchmark configuration as an input, and starts sending HTTP requests to orchestrator to run GPU functions.

We will use the CUDA example of vector addition.

```
${BUILD_DIR}/invoker/bin/invoker ${BUILD_DIR}/examples/vector_add.json result.csv
```

