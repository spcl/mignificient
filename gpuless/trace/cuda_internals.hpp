#ifndef GPULESS_CUDA_INTERNALS_HPP
#define GPULESS_CUDA_INTERNALS_HPP

extern "C" {
extern void **__cudaRegisterFatBinary(void *fatCubin);
extern void __cudaRegisterFatBinaryEnd(void **fatCubinHandle);
extern void __cudaUnregisterFatBinary(void **fatCubinHandle);
extern void CUDARTAPI __cudaRegisterFunction(
    void **fatCubinHandle, const char *hostFun, char *deviceFun,
    const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid,
    dim3 *bDim, dim3 *gDim, int *wSize);
}

#endif // GPULESS_CUDA_INTERNALS_HPP
