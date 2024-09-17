# mignificient



# Dependencies

flatbuffers

libbibery

## Example of building on cluster

```
pybind11_DIR=/users/mcopik/anaconda3/lib/python3.12/site-packages/pybind11 cmake -DCUDNN_DIR=/scratch/mcopik/gpus/cudnn-8/ -DCMAKE_C_FLAGS="-I ${DEPS_PATH}/include" -DCMAKE_CXX_FLAGS="-I ${DEPS_PATH}/include" -DCMAKE_CXX_STANDARD_LIBRARIES="-L${DEPS_PATH}/lib"  -DCMAKE_BUILD_TYPE=Debug ../
```
