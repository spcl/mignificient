# mignificient



# Dependencies

CUDA 11.6

cuDNN 8.9.7 for CUDA 11

libacl - if these are not available on your system (usually visible through compilation errors caused by missing headers) and you can't install them as package, then install them locally in `DEPS_PATH`.

https://download.savannah.nongnu.org/releases/acl/acl-2.3.2.tar.xz

pybind11

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

