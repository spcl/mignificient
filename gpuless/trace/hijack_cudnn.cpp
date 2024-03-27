#include <cudnn.h>
#include <numeric>

#include "cudnn_api_calls.hpp"
#include "libgpuless.hpp"

extern "C" {

static uint64_t nextCudnnHandle() {
    static uint64_t next = 1;
    return next++;
}

static uint64_t nextCudnnTensorDescriptor() {
    static uint64_t next = 1;
    return next++;
}

static uint64_t nextCudnnFilterDescriptor() {
    static uint64_t next = 1;
    return next++;
}

static uint64_t nextCudnnConvolutionDescriptor() {
    static uint64_t next = 1;
    return next++;
}

static std::vector<size_t> &getTensorDescriptorToSize() {
    static std::vector<size_t> virtual_cd_to_size;
    return virtual_cd_to_size;
}

static void recordTensorDescriptorSize(uint64_t virutal_cd,
                                       cudnnDataType_t data_type) {
    auto &virtual_cd_to_size = getTensorDescriptorToSize();
    if (virtual_cd_to_size.size() < virutal_cd + 1) {
        virtual_cd_to_size.resize(virutal_cd + 1);
    }
    size_t type_size;
    if (data_type == CUDNN_DATA_DOUBLE) {
        type_size = sizeof(double);
    } else {
        type_size = sizeof(float);
    }
    virtual_cd_to_size[virutal_cd] = type_size;
}

cudnnStatus_t cudnnCreate(cudnnHandle_t *handle) {
    HIJACK_FN_PROLOGUE();
    auto virtual_handle = nextCudnnHandle();
    *handle = reinterpret_cast<cudnnHandle_t>(virtual_handle);
    getCudaTrace().record(
        std::make_shared<gpuless::CudnnCreate>(virtual_handle));
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId) {
    HIJACK_FN_PROLOGUE();
    getCudaTrace().record(std::make_shared<gpuless::CudnnSetStream>(
        reinterpret_cast<uint64_t>(handle), streamId));
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc) {
    HIJACK_FN_PROLOGUE();
    auto next_desc = nextCudnnTensorDescriptor();
    getCudaTrace().record(
        std::make_shared<gpuless::CudnnCreateTensorDescriptor>(next_desc));
    *tensorDesc = reinterpret_cast<cudnnTensorDescriptor_t>(next_desc);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                         cudnnDataType_t dataType, int nbDims,
                                         const int dimA[],
                                         const int strideA[]) {
    HIJACK_FN_PROLOGUE();
    std::vector<int> dim_a_vec(nbDims);
    std::vector<int> stride_a_vec(nbDims);
    std::memcpy(dim_a_vec.data(), dimA, nbDims * sizeof(int));
    std::memcpy(stride_a_vec.data(), strideA, nbDims * sizeof(int));

    auto virtual_td = reinterpret_cast<uint64_t>(tensorDesc);
    recordTensorDescriptorSize(virtual_td, dataType);

    getCudaTrace().record(std::make_shared<gpuless::CudnnSetTensorNdDescriptor>(
        virtual_td, dataType, nbDims, dim_a_vec, stride_a_vec));
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc) {
    HIJACK_FN_PROLOGUE();
    auto next_desc = nextCudnnFilterDescriptor();
    getCudaTrace().record(
        std::make_shared<gpuless::CudnnCreateFilterDescriptor>(next_desc));
    *filterDesc = reinterpret_cast<cudnnFilterDescriptor_t>(next_desc);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t filterDesc,
                                         cudnnDataType_t dataType,
                                         cudnnTensorFormat_t format, int nbDims,
                                         const int filterDimA[]) {
    HIJACK_FN_PROLOGUE();
    std::vector<int> filter_dim_a_vec(nbDims);
    std::memcpy(filter_dim_a_vec.data(), filterDimA, nbDims * sizeof(int));
    getCudaTrace().record(std::make_shared<gpuless::CudnnSetFilterNdDescriptor>(
        reinterpret_cast<uint64_t>(filterDesc), dataType, format, nbDims,
        filter_dim_a_vec));
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t
cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc) {
    HIJACK_FN_PROLOGUE();
    auto next_desc = nextCudnnConvolutionDescriptor();
    getCudaTrace().record(
        std::make_shared<gpuless::CudnnCreateConvolutionDescriptor>(next_desc));
    *convDesc = reinterpret_cast<cudnnConvolutionDescriptor_t>(next_desc);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t
cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc,
                              int groupCount) {
    HIJACK_FN_PROLOGUE();
    getCudaTrace().record(
        std::make_shared<gpuless::CudnnSetConvolutionGroupCount>(
            reinterpret_cast<uint64_t>(convDesc), groupCount));
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc,
                                          cudnnMathType_t mathType) {
    HIJACK_FN_PROLOGUE();
    getCudaTrace().record(
        std::make_shared<gpuless::CudnnSetConvolutionMathType>(
            reinterpret_cast<uint64_t>(convDesc), mathType));
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionNdDescriptor(
    cudnnConvolutionDescriptor_t convDesc, int arrayLength, const int padA[],
    const int filterStrideA[], const int dilationA[],
    cudnnConvolutionMode_t mode, cudnnDataType_t dataType) {
    HIJACK_FN_PROLOGUE();
    std::vector<int> pad_a_vec(arrayLength);
    std::vector<int> filter_stride_a_vec(arrayLength);
    std::vector<int> dilation_a_vec(arrayLength);
    std::memcpy(pad_a_vec.data(), padA, arrayLength * sizeof(int));
    std::memcpy(filter_stride_a_vec.data(), filterStrideA,
                arrayLength * sizeof(int));
    std::memcpy(dilation_a_vec.data(), dilationA, arrayLength * sizeof(int));
    getCudaTrace().record(
        std::make_shared<gpuless::CudnnSetConvolutionNdDescriptor>(
            reinterpret_cast<uint64_t>(convDesc), arrayLength, pad_a_vec,
            filter_stride_a_vec, dilation_a_vec, mode, dataType));
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults) {
    HIJACK_FN_PROLOGUE();
    getCudaTrace().record(
        std::make_shared<gpuless::CudnnGetConvolutionForwardAlgorithmV7>(
            reinterpret_cast<uint64_t>(handle),
            reinterpret_cast<uint64_t>(xDesc),
            reinterpret_cast<uint64_t>(yDesc),
            reinterpret_cast<uint64_t>(wDesc),
            reinterpret_cast<uint64_t>(convDesc), requestedAlgoCount));
    getTraceExecutor()->synchronize(getCudaTrace());
    auto top = std::static_pointer_cast<
        gpuless::CudnnGetConvolutionForwardAlgorithmV7>(
        getCudaTrace().historyTop());
    *returnedAlgoCount = top->returned_algo_count;
    std::memcpy(perfResults, top->perf_results.data(),
                top->returned_algo_count *
                    sizeof(cudnnConvolutionFwdAlgoPerf_t));
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t
cudnnConvolutionForward(cudnnHandle_t handle, const void *alpha,
                        const cudnnTensorDescriptor_t xDesc, const void *x,
                        const cudnnFilterDescriptor_t wDesc, const void *w,
                        const cudnnConvolutionDescriptor_t convDesc,
                        cudnnConvolutionFwdAlgo_t algo, void *workSpace,
                        size_t workSpaceSizeInBytes, const void *beta,
                        const cudnnTensorDescriptor_t yDesc, void *y) {
    HIJACK_FN_PROLOGUE();

    // get size of alpha, beta in bytes
    auto virtual_td_y = reinterpret_cast<uint64_t>(yDesc);
    auto &td_to_size = getTensorDescriptorToSize();
    size_t scaling_size = td_to_size[virtual_td_y];

    getCudaTrace().record(std::make_shared<gpuless::CudnnConvolutionForward>(
        reinterpret_cast<uint64_t>(handle), scaling_size, alpha, beta,
        workSpace, workSpaceSizeInBytes, reinterpret_cast<uint64_t>(convDesc),
        algo, reinterpret_cast<uint64_t>(wDesc), w,
        reinterpret_cast<uint64_t>(xDesc), x, reinterpret_cast<uint64_t>(yDesc),
        y));
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBackwardData(
    cudnnHandle_t handle, const void *alpha,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {
    HIJACK_FN_PROLOGUE();

    std::vector<uint8_t> alpha_vec(sizeof(float));
    std::vector<uint8_t> beta_vec(sizeof(float));
    std::memcpy(alpha_vec.data(), alpha, sizeof(float));
    std::memcpy(beta_vec.data(), beta, sizeof(float));

    getCudaTrace().record(
        std::make_shared<gpuless::CudnnConvolutionBackwardData>(
            reinterpret_cast<uint64_t>(handle), alpha_vec,
            reinterpret_cast<uint64_t>(wDesc), w,
            reinterpret_cast<uint64_t>(dyDesc), dy,
            reinterpret_cast<uint64_t>(convDesc), algo, workSpace,
            workSpaceSizeInBytes, beta_vec, reinterpret_cast<uint64_t>(dxDesc),
            dx));
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults) {
    HIJACK_FN_PROLOGUE();

    getCudaTrace().record(
        std::make_shared<gpuless::CudnnGetConvolutionBackwardDataAlgorithmV7>(
            reinterpret_cast<uint64_t>(handle),
            reinterpret_cast<uint64_t>(wDesc),
            reinterpret_cast<uint64_t>(dyDesc),
            reinterpret_cast<uint64_t>(convDesc),
            reinterpret_cast<uint64_t>(dxDesc), requestedAlgoCount));

    getTraceExecutor()->synchronize(getCudaTrace());
    auto top = std::static_pointer_cast<
        gpuless::CudnnGetConvolutionBackwardDataAlgorithmV7>(
        getCudaTrace().historyTop());
    *returnedAlgoCount = top->returned_algo_count;
    std::memcpy(perfResults, top->perf_results.data(),
                top->returned_algo_count *
                    sizeof(cudnnConvolutionFwdAlgoPerf_t));

    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t zDesc,
    const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const cudnnActivationDescriptor_t activationDesc, size_t *sizeInBytes) {
    HIJACK_FN_PROLOGUE();

    getCudaTrace().record(
        std::make_shared<
            gpuless::CudnnGetBatchNormalizationForwardTrainingExWorkspaceSize>(
            reinterpret_cast<uint64_t>(handle), mode, bnOps,
            reinterpret_cast<uint64_t>(xDesc),
            reinterpret_cast<uint64_t>(zDesc),
            reinterpret_cast<uint64_t>(yDesc),
            reinterpret_cast<uint64_t>(bnScaleBiasMeanVarDesc),
            reinterpret_cast<uint64_t>(activationDesc)));
    getTraceExecutor()->synchronize(getCudaTrace());
    auto top = std::static_pointer_cast<
        gpuless::CudnnGetBatchNormalizationForwardTrainingExWorkspaceSize>(
        getCudaTrace().historyTop());
    *sizeInBytes = top->size_in_bytes;

    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes) {
    HIJACK_FN_PROLOGUE();

    getCudaTrace().record(
        std::make_shared<
            gpuless::CudnnGetBatchNormalizationTrainingExReserveSpaceSize>(
            reinterpret_cast<uint64_t>(handle), mode, bnOps,
            reinterpret_cast<uint64_t>(activationDesc),
            reinterpret_cast<uint64_t>(xDesc)));
    getTraceExecutor()->synchronize(getCudaTrace());
    auto top = std::static_pointer_cast<
        gpuless::CudnnGetBatchNormalizationTrainingExReserveSpaceSize>(
        getCudaTrace().historyTop());
    *sizeInBytes = top->size_in_bytes;
    SPDLOG_DEBUG("cudnnGetBatchNormalizationTrainingExReserveSpaceSize() "
                  "[sizeInBytes={}]",
                  *sizeInBytes);

    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationForwardInference(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha,
    const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, const void *estimatedMean,
    const void *estimatedVariance, double epsilon) {
    HIJACK_FN_PROLOGUE();

    // get size of alpha, beta in bytes
    auto virtual_td_y = reinterpret_cast<uint64_t>(yDesc);
    auto &td_to_size = getTensorDescriptorToSize();
    size_t scaling_size = td_to_size[virtual_td_y];

    getCudaTrace().record(
        std::make_shared<gpuless::CudnnBatchNormalizationForwardInference>(
            reinterpret_cast<uint64_t>(handle), mode, scaling_size, alpha, beta,
            reinterpret_cast<uint64_t>(xDesc), x,
            reinterpret_cast<uint64_t>(yDesc), y,
            reinterpret_cast<uint64_t>(bnScaleBiasMeanVarDesc), bnScale, bnBias,
            estimatedMean, estimatedVariance, epsilon));

    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc,
    const void *xData, const cudnnTensorDescriptor_t zDesc, const void *zData,
    const cudnnTensorDescriptor_t yDesc, void *yData,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void *bnScaleData, const void *bnBiasData,
    double exponentialAverageFactor, void *resultRunningMeanData,
    void *resultRunningVarianceData, double epsilon, void *saveMean,
    void *saveInvVariance, const cudnnActivationDescriptor_t activationDesc,
    void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
    HIJACK_FN_PROLOGUE();

    std::vector<uint8_t> alpha_vec(sizeof(float));
    std::vector<uint8_t> beta_vec(sizeof(float));
    std::memcpy(alpha_vec.data(), alpha, sizeof(float));
    std::memcpy(beta_vec.data(), beta, sizeof(float));

    getCudaTrace().record(
        std::make_shared<gpuless::CudnnBatchNormalizationForwardTrainingEx>(
            reinterpret_cast<uint64_t>(handle), mode, bnOps, alpha_vec,
            beta_vec, reinterpret_cast<uint64_t>(xDesc), xData,
            reinterpret_cast<uint64_t>(yDesc), yData,
            reinterpret_cast<uint64_t>(zDesc), zData,
            reinterpret_cast<uint64_t>(bnScaleBiasMeanVarDesc), bnScaleData,
            bnBiasData, exponentialAverageFactor, resultRunningMeanData,
            resultRunningVarianceData, epsilon, saveMean, saveInvVariance,
            reinterpret_cast<uint64_t>(activationDesc), workspace,
            workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes));

    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t
cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc) {
    HIJACK_FN_PROLOGUE();
    getCudaTrace().record(
        std::make_shared<gpuless::CudnnDestroyConvolutionDescriptor>(
            reinterpret_cast<uint64_t>(convDesc)));
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc) {
    HIJACK_FN_PROLOGUE();
    getCudaTrace().record(
        std::make_shared<gpuless::CudnnDestroyFilterDescriptor>(
            reinterpret_cast<uint64_t>(filterDesc)));
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {
    HIJACK_FN_PROLOGUE();
    getCudaTrace().record(
        std::make_shared<gpuless::CudnnDestroyTensorDescriptor>(
            reinterpret_cast<uint64_t>(tensorDesc)));
    return CUDNN_STATUS_SUCCESS;
}
}