#include "cuda_virtual_device.hpp"
#include <spdlog/spdlog.h>

void CudaVirtualDevice::initRealDevice() {
    if (this->initialized) {
        return;
    }

    SPDLOG_INFO("CudaVirtualDevice: initializing real device");
    checkCudaErrors(cuInit(0));
    checkCudaErrors(cuDeviceGet(&this->device, 0));
    checkCudaErrors(cuCtxCreate(&this->context, 0, device));
    checkCudaErrors(cuCtxSetCurrent(this->context));
    checkCudaErrors(cuDevicePrimaryCtxRetain(&this->context, this->device));

    size_t n_attributes = CU_DEVICE_ATTRIBUTE_MAX;
    this->device_attributes.resize(n_attributes);
    for (size_t i = 1; i < n_attributes; i++) {
        checkCudaErrors(cuDeviceGetAttribute(&this->device_attributes[i],
                                             static_cast<CUdevice_attribute>(i),
                                             this->device));
        SPDLOG_TRACE("Device attribute {}: {}", i, this->device_attributes[i]);
    }

    checkCudaErrors(cuDeviceTotalMem(&this->device_total_mem, this->device));
    SPDLOG_TRACE("Device TotalMem: {}", this->device_total_mem);

    this->initialized = true;
}
void *CudaVirtualDevice::get_scratch(size_t size) {
    if (scratch_used)
        throw std::runtime_error("Scratch already in use.");
    scratch_used = true;

    if (size < scratch_size)
        return scratch_memory;

    if (scratch_memory != nullptr)
        cudaFree(scratch_memory);

    cudaMalloc(&scratch_memory, size);
    return scratch_memory;
}
void CudaVirtualDevice::free_scratch() { scratch_used = false; }
