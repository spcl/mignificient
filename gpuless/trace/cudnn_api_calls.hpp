#ifndef GPULESS_CUDNN_API_CALLS_HPP
#define GPULESS_CUDNN_API_CALLS_HPP

#include "cuda_api_calls.hpp"
#include <cstdint>

namespace gpuless {

class CudaCudnnApiCall : public AbstractCudaApiCall {
  public:
    std::string nativeErrorToString(uint64_t err) override;
};

class CudnnCreate : public CudaCudnnApiCall {
  public:
    uint64_t virtual_handle;

    explicit CudnnCreate(uint64_t virtualHandle);
    explicit CudnnCreate(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudnnSetStream : public CudaCudnnApiCall {
  public:
    uint64_t virtual_handle;
    cudaStream_t stream;

    CudnnSetStream(uint64_t virtualHandle, cudaStream_t stream);
    explicit CudnnSetStream(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudnnCreateTensorDescriptor : public CudaCudnnApiCall {
  public:
    uint64_t virtual_td;

    explicit CudnnCreateTensorDescriptor(uint64_t virtualTd);
    explicit CudnnCreateTensorDescriptor(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudnnSetTensorNdDescriptor : public CudaCudnnApiCall {
  public:
    uint64_t virtual_td;
    cudnnDataType_t data_type;
    int nb_dims;
    std::vector<int> dim_a;
    std::vector<int> stride_a;

    CudnnSetTensorNdDescriptor(uint64_t virtualTd, cudnnDataType_t dataType,
                               int nbDims, std::vector<int> dimA,
                               std::vector<int> strideA);
    explicit CudnnSetTensorNdDescriptor(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudnnCreateFilterDescriptor : public CudaCudnnApiCall {
  public:
    uint64_t virtual_fd;

    explicit CudnnCreateFilterDescriptor(uint64_t virtualFd);
    explicit CudnnCreateFilterDescriptor(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudnnSetFilterNdDescriptor : public CudaCudnnApiCall {
  public:
    uint64_t virtual_fd;
    cudnnDataType_t data_type;
    cudnnTensorFormat_t format;
    int nb_dims;
    std::vector<int> filter_dim_a;

    CudnnSetFilterNdDescriptor(uint64_t virtualFd, cudnnDataType_t dataType,
                               cudnnTensorFormat_t format, int nbDims,
                               const std::vector<int> &filterDimA);
    explicit CudnnSetFilterNdDescriptor(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudnnCreateConvolutionDescriptor : public CudaCudnnApiCall {
  public:
    uint64_t virtual_cd;

    explicit CudnnCreateConvolutionDescriptor(uint64_t virtualCd);
    explicit CudnnCreateConvolutionDescriptor(
        const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudnnSetConvolutionGroupCount : public CudaCudnnApiCall {
  public:
    uint64_t virtual_cd;
    int group_count;

    CudnnSetConvolutionGroupCount(uint64_t virtualCd, int groupCount);
    explicit CudnnSetConvolutionGroupCount(
        const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudnnSetConvolutionMathType : public CudaCudnnApiCall {
  public:
    uint64_t virtual_cd;
    cudnnMathType_t math_type;

    CudnnSetConvolutionMathType(uint64_t virtualCd, cudnnMathType_t mathType);
    explicit CudnnSetConvolutionMathType(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudnnSetConvolutionNdDescriptor : public CudaCudnnApiCall {
  public:
    uint64_t virtual_cd;
    int array_length;
    std::vector<int> pad_a;
    std::vector<int> filter_stride_a;
    std::vector<int> dilation;
    cudnnConvolutionMode_t convolution_mode;
    cudnnDataType_t cudnn_data_type;

    CudnnSetConvolutionNdDescriptor(uint64_t virtualCd, int arrayLength,
                                    std::vector<int> padA,
                                    std::vector<int> filterStrideA,
                                    std::vector<int> dilation,
                                    cudnnConvolutionMode_t convolutionMode,
                                    cudnnDataType_t cudnnDataType);
    explicit CudnnSetConvolutionNdDescriptor(
        const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudnnGetConvolutionForwardAlgorithmV7 : public CudaCudnnApiCall {
  public:
    uint64_t virtual_handle;
    uint64_t virtual_td_xdesc;
    uint64_t virtual_td_ydesc;
    uint64_t virtual_fd;
    uint64_t virtual_cd;
    int requested_algo_count;

    // outputs
    int returned_algo_count{};
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results;

    CudnnGetConvolutionForwardAlgorithmV7(uint64_t virtualHandle,
                                          uint64_t virtualTdXdesc,
                                          uint64_t virtualTdYdesc,
                                          uint64_t virtualFd,
                                          uint64_t virtualCd,
                                          int requestedAlgoCount);
    explicit CudnnGetConvolutionForwardAlgorithmV7(
        const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudnnConvolutionForward : public CudaCudnnApiCall {
  public:
    uint64_t virtual_handle;
    std::vector<uint8_t> alpha;
    std::vector<uint8_t> beta;
    void *workspace;
    size_t workspace_size_in_bytes;
    uint64_t virtual_cd;
    cudnnConvolutionFwdAlgo_t algo;
    uint64_t virtual_fd_wdesc;
    const void *w;
    uint64_t virtual_td_xdesc;
    const void *x;
    uint64_t virtual_td_ydesc;
    void *y;

    CudnnConvolutionForward(uint64_t virtualHandle, size_t scaling_size,
                            const void *alpha_ptr, const void *beta_ptr,
                            void *workspace, size_t workspaceSizeInBytes,
                            uint64_t virtualCd, cudnnConvolutionFwdAlgo_t algo,
                            uint64_t virtualFdWdesc, const void *w,
                            uint64_t virtualTdXdesc, const void *x,
                            uint64_t virtualTdYdesc, void *y);
    explicit CudnnConvolutionForward(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudnnBatchNormalizationForwardInference : public CudaCudnnApiCall {
  public:
    uint64_t virtual_handle;
    cudnnBatchNormMode_t mode;
    std::vector<uint8_t> alpha;
    std::vector<uint8_t> beta;
    uint64_t virtual_td_xdesc;
    const void *x;
    uint64_t virtual_td_ydesc;
    void *y;

    uint64_t virtual_td_bs_scale_bias_mean_var_desc;
    const void *bn_scale;
    const void *bn_bias;
    const void *estimated_mean;
    const void *estimated_variance;
    double epsilon;

    CudnnBatchNormalizationForwardInference(
        uint64_t virtualHandle, cudnnBatchNormMode_t mode, size_t scaling_size,
        const void *alpha_ptr, const void *beta_ptr, uint64_t virtualTdXdesc,
        const void *x, uint64_t virtualTdYdesc, void *y,
        uint64_t virtualTdBsScaleBiasMeanVarDesc, const void *bnScale,
        const void *bnBias, const void *estimatedMean,
        const void *estimatedVariance, double epsilon);
    explicit CudnnBatchNormalizationForwardInference(
        const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudnnDestroyConvolutionDescriptor : public CudaCudnnApiCall {
  public:
    uint64_t virtual_cd;
    explicit CudnnDestroyConvolutionDescriptor(uint64_t virtualCd);
    explicit CudnnDestroyConvolutionDescriptor(
        const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudnnDestroyFilterDescriptor : public CudaCudnnApiCall {
  public:
    uint64_t virtual_fd;

    explicit CudnnDestroyFilterDescriptor(uint64_t virtualFd);
    explicit CudnnDestroyFilterDescriptor(
        const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudnnDestroyTensorDescriptor : public CudaCudnnApiCall {
  public:
    uint64_t virtual_td;
    explicit CudnnDestroyTensorDescriptor(uint64_t virtualTd);
    explicit CudnnDestroyTensorDescriptor(
        const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudnnConvolutionBackwardData : public CudaCudnnApiCall {
  public:
    uint64_t virtual_handle;
    std::vector<uint8_t> alpha;
    uint64_t virtual_fd_wdesc;
    const void *w;
    uint64_t virtual_td_dydesc;
    const void *dy;
    uint64_t virtual_cd;
    cudnnConvolutionBwdDataAlgo_t algo;
    void *workspace;
    size_t workspace_size_in_bytes;
    std::vector<uint8_t> beta;
    uint64_t virtual_td_dxdesc;
    void *dx;

    CudnnConvolutionBackwardData(const uint64_t &virtualHandle,
                                 const std::vector<uint8_t> &alpha,
                                 const uint64_t &virtualFdWdesc, const void *w,
                                 const uint64_t &virtualTdDydesc,
                                 const void *dy, const uint64_t &virtualCd,
                                 cudnnConvolutionBwdDataAlgo_t algo,
                                 void *workspace, size_t workspaceSizeInBytes,
                                 const std::vector<uint8_t> &beta,
                                 const uint64_t &virtualTdDxdesc, void *dx);
    explicit CudnnConvolutionBackwardData(
        const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;
    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudnnGetConvolutionBackwardDataAlgorithmV7 : public CudaCudnnApiCall {
  public:
    uint64_t virtual_handle;
    uint64_t virtual_fd_wdesc;
    uint64_t virtual_td_dydesc;
    uint64_t virtual_cd_convdesc;
    uint64_t virtual_td_dxdesc;
    int requested_algo_count;

    // outputs
    int returned_algo_count{};
    std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results;

    CudnnGetConvolutionBackwardDataAlgorithmV7(uint64_t virtualHandle,
                                               uint64_t virtualFdWdesc,
                                               uint64_t virtualTdDydesc,
                                               uint64_t virtualCdConvdesc,
                                               uint64_t virtualTdDxdesc,
                                               int requestedAlgoCount);
    explicit CudnnGetConvolutionBackwardDataAlgorithmV7(
        const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;
    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudnnGetBatchNormalizationForwardTrainingExWorkspaceSize
    : public CudaCudnnApiCall {
  public:
    uint64_t virtual_handle;
    cudnnBatchNormMode_t mode;
    cudnnBatchNormOps_t bn_ops;
    uint64_t virtual_td_xdesc;
    uint64_t virtual_td_zdesc;
    uint64_t virtual_td_ydesc;
    uint64_t virtual_td_bn_scale_bias_mean_var_desc;
    uint64_t virtual_ad_activation_desc;

    // output
    size_t size_in_bytes{};

    CudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
        uint64_t virtualHandle, cudnnBatchNormMode_t mode,
        cudnnBatchNormOps_t bnOps, uint64_t virtualTdXdesc,
        uint64_t virtualTdZdesc, uint64_t virtualTdYdesc,
        uint64_t virtualTdBnScaleBiasMeanVarDesc,
        uint64_t virtualAdActivationDesc);
    explicit CudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
        const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudnnGetBatchNormalizationTrainingExReserveSpaceSize
    : public CudaCudnnApiCall {
  public:
    uint64_t virtual_handle;
    cudnnBatchNormMode_t mode;
    cudnnBatchNormOps_t bn_ops;
    uint64_t virtual_ad_activation_desc;
    uint64_t virtual_td_xdesc;

    // output
    size_t size_in_bytes{};

    CudnnGetBatchNormalizationTrainingExReserveSpaceSize(
        uint64_t virtualHandle, cudnnBatchNormMode_t mode,
        cudnnBatchNormOps_t bnOps, uint64_t virtualAdActivationDesc,
        uint64_t virtualTdXdesc);
    explicit CudnnGetBatchNormalizationTrainingExReserveSpaceSize(
        const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CudnnBatchNormalizationForwardTrainingEx : public CudaCudnnApiCall {
  public:
    uint64_t virtual_handle;
    cudnnBatchNormMode_t mode;
    cudnnBatchNormOps_t bn_ops;
    std::vector<uint8_t> alpha;
    std::vector<uint8_t> beta;
    uint64_t virtual_td_xdesc;
    const void *x_data;
    uint64_t virtual_td_ydesc;
    void *y_data;
    uint64_t virtual_td_zdesc;
    const void *z_data;
    uint64_t virtual_td_bn_scale_bias_mean_var_desc;
    const void *bn_scale_data;
    const void *bn_bias_data;
    double exponential_average_factor;
    void *result_running_mean_data;
    void *result_running_variance_data;
    double epsilon;
    void *save_mean;
    void *save_inv_variance;
    uint64_t virtual_ad_activation_desc;
    void *workspace;
    size_t workspace_size_in_bytes;
    void *reserve_space;
    size_t reserve_space_size_in_bytes;

    CudnnBatchNormalizationForwardTrainingEx(
        uint64_t virtualHandle, cudnnBatchNormMode_t mode,
        cudnnBatchNormOps_t bnOps, std::vector<uint8_t> &alpha,
        std::vector<uint8_t> &beta, uint64_t virtualTdXdesc,
        const void *xData, uint64_t virtualTdYdesc, void *yData,
        uint64_t virtualTdZdesc, const void *zData,
        uint64_t virtualTdBnScaleBiasMeanVarDesc, const void *bnScaleData,
        const void *bnBiasData, double exponentialAverageFactor,
        void *resultRunningMeanData, void *resultRunningVarianceData,
        double epsilon, void *saveMean, void *saveInvVariance,
        uint64_t virtualAdActivationDesc, void *workspace,
        size_t workspaceSizeInBytes, void *reserveSpace,
        size_t reserveSpaceSizeInBytes);
    explicit CudnnBatchNormalizationForwardTrainingEx(
        const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

} // namespace gpuless

#endif // GPULESS_CUDNN_API_CALLS_HPP
