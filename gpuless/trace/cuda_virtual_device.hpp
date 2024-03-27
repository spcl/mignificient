#ifndef GPULESS_CUDA_VDEV_H
#define GPULESS_CUDA_VDEV_H

#include "../utils.hpp"
#include <cstdint>
#include <cublas.h>
#include <cublasLt.h>
#include <cuda.h>
#include <cudnn.h>
#include <fatbinary_section.h>
#include <iostream>
#include <map>
#include <string>

class CudaVirtualDevice {
  private:
    bool initialized = false;
    void *scratch_memory = nullptr;
    size_t scratch_size = 0;
    bool scratch_used = false;

  public:
    // virtualization of cudnn handles to avoid costly synchronizations
    std::vector<cudnnHandle_t> cudnn_handles_virtual_to_real;
    std::vector<cudnnTensorDescriptor_t>
        cudnn_tensor_descriptor_virtual_to_real;
    std::vector<cudnnFilterDescriptor_t>
        cudnn_filter_descriptor_virtual_to_real;
    std::vector<cudnnConvolutionDescriptor_t>
        cudnn_convolution_descriptor_virtual_to_real;

    // virtualization of cudnn algorithms
    std::vector<cudnnConvolutionFwdAlgoPerf_t>
        cudnn_convolution_fwd_algo_perf_virtual_to_real;
    std::vector<cudnnConvolutionBwdDataAlgoPerf_t>
        cudnn_convolution_bwd_data_algo_perf_virtual_to_real;

    // virtualization of cublas handles to avoid costly synchronizations
    std::vector<cublasHandle_t> cublas_handle_virtual_to_real;
    std::vector<cublasLtHandle_t> cublaslt_handle_virtual_to_real;
    std::vector<cublasLtMatmulDesc_t> cublaslt_matmul_handle_virtual_to_real;
    std::vector<cublasLtMatrixLayout_t>
        cublaslt_matrix_layout_handle_virtual_to_real;
    std::vector<cublasLtMatmulPreference_t>
        cublaslt_matmul_pref_handle_virtual_to_real;
    std::vector<cublasLtMatmulHeuristicResult_t >
        cublaslt_matmul_alg_virtual_to_real;

    // stored device attributes
    size_t device_total_mem = 0;
    std::vector<int32_t> device_attributes;

    std::map<uint64_t, CUmodule> module_registry_;
    std::map<std::string, CUfunction> function_registry_;
    CUdevice device;
    CUcontext context;

    void initRealDevice();

    void *get_scratch(size_t size);
    void free_scratch();

    ~CudaVirtualDevice() {
        if (scratch_memory != nullptr)
            cudaFree(scratch_memory);
    }
};

#endif // GPULESS_CUDA_VDEV_H
