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
export PATH=$CUDA/bin:$PATH
cmake -S . -B build-cu116 -DCMAKE_BUILD_TYPE=Release -DBUILD_ICEORYX2=TRUE -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_COMPILER=$CUDA/bin/nvcc -DCUDAToolkit_ROOT=$CUDA \
  -DCUDNN_DIR=<cudnn-8.9.7-for-cuda11-archive>/ \
  -Dpybind11_DIR=<site-packages>/pybind11/share/cmake/pybind11 \
  -DPython_EXECUTABLE=<python3.9-env>/bin/python
cmake --build build-cu116 -j16
```

The configure step fetches and builds iceoryx2 (needs git and cargo): the version in `external/iceoryx2-version` with `external/iceoryx2-file-backend.patch`.
To use a prebuilt one instead, install it with `external/iceoryx2-build.sh <src> <build> <prefix>` (same version and patch) and pass `-DMIGNIFICIENT_ICEORYX2_INSTALL=<prefix>`; only that prefix is searched, never a system iceoryx2.
With the patch, its `ipc` service keeps data segments, dynamic configs and connections as files under the config's `global.root-path` instead of POSIX shared memory in `/dev/shm`.
The orchestrator gives every client its own root, `<ipc.client-root-base>/<16 hex digits of a hash of the client id>` (mode 0700; default base `/dev/shm/mignificient`, a tmpfs, at most 39 characters because the event sockets in the root must fit a 107-byte unix socket path). The orchestrator log prints each client's directory. The gpuless server and the executor get the root in `MIGNIFICIENT_IOX2_ROOT`; gpuless also keeps its memcpy chunks there. A container mounts only that directory (plus the iceoryx2 config file, read-only), and the directory is removed when the client goes away (timeout, OOM, startup failure, admin kill, orchestrator shutdown).
All processes load the iceoryx2 config from `ipc.iceoryx2-config` (default: `config/iceoryx2.toml`, copied to `<build>/config/`), passed on in `MIGNIFICIENT_IOX2_CONFIG`; without the key iceoryx2 searches its default locations (`./config/`, `~/.config/iceoryx2/`, `/etc/iceoryx2/`).
The executor image must be built from the same patched iceoryx2 (`docker/Dockerfile.iceoryx` runs the same script). A process built against unpatched iceoryx2 looks for the data in `/dev/shm` and never finds the other side's services.

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

### Start Iceoryx's Roudi (iceoryx1 backend only)

Only needed with `"ipc": {"backend": "iceoryx1"}`; the default iceoryx2 backend has no central daemon.
Start it once per node; in case of issues with iceoryx, kill the `iox-roudi` process and start it again.

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

### Function packages and containers

A request runs either bare-metal (the default, `executor.type` in the config) or in a container; the first request
of a client may choose with `"executor": "bare-metal" | "container"`. Containers need a function package, passed as
`"code-package": "<absolute path>"`, which must lie under one of the config's `executor.package-roots`; the
request's `function-path` (and `cubin-analysis`, if given) must be inside it. The container mounts the package
read-only at the same path, plus its own iceoryx2 directory.

Packages are built with `tools/pack_code.py`:

```
# python: copies the code, creates a conda env (PyTorch 1.12.1, CUDA 11.6), runs prepare.py (e.g. downloads
# weights into torch-cache/), and writes the cubin analysis gpuless needs for the env's CUDA libraries
tools/pack_code.py <function dir> python --dest <packages dir> --cubin-analyzer ${BUILD_DIR}/gpuless/cubin_analyzer
# C++: code and the function's .so
tools/pack_code.py <function dir> cpp --dest <packages dir> --so libfunction.so
```

`package.json` in the package records the language, runtime, `torch-home` and `cubin-analysis`; a request without
`cubin-analysis` uses the package's. See the docstring of `tools/pack_code.py` for details.
The executor image: `docker build -f docker/Dockerfile.iceoryx -t spcleth/mignificient:iceoryx-local .`, then
`docker build -f docker/Dockerfile --build-arg ICEORYX_IMAGE=spcleth/mignificient:iceoryx-local -t spcleth/mignificient:executor-local .`
(`executor.container-executor.image` in the config). Pull or build it on every node before starting the orchestrator.

### Tests

`ctest -L integration --output-on-failure` in the build directory (needs a GPU; the ResNet-50 tests pack a package
into `<build>/test-packages` once, which needs network and conda). Manual failure scenarios (startup failures, OOM,
crashed gpuless server, per-client isolation) are in `tests/manual/`.

### Start invoker

This test processes takes a benchmark configuration as an input, and starts sending HTTP requests to orchestrator to run GPU functions.

We will use the CUDA example of vector addition.

```
${BUILD_DIR}/invoker/bin/invoker ${BUILD_DIR}/examples/vector_add.json result.csv
```

