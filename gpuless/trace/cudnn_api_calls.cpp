#include "cudnn_api_calls.hpp"
#include "dlsym_util.hpp"
#include <cudnn.h>
#include <dlfcn.h>

#include "libgpuless.hpp"
#include <cstdint>
#include <utility>

namespace gpuless {

std::string gpuless::CudaCudnnApiCall::nativeErrorToString(uint64_t err) {
    auto str =
        "[cudnn] " +
        std::string(cudnnGetErrorString(static_cast<cudnnStatus_t>(err)));
    return str;
}

/*
 * cudnnCreate
 */
uint64_t gpuless::CudnnCreate::executeNative(CudaVirtualDevice &vdev) {
    static auto real =
        (decltype(&cudnnCreate))real_dlsym(RTLD_NEXT, "cudnnCreate");
    if (vdev.cudnn_handles_virtual_to_real.size() < this->virtual_handle + 1) {
        vdev.cudnn_handles_virtual_to_real.resize(this->virtual_handle + 1);
    }
    return real(&vdev.cudnn_handles_virtual_to_real[this->virtual_handle]);
}

gpuless::CudnnCreate::CudnnCreate(uint64_t virtualHandle)
    : virtual_handle(virtualHandle) {}

flatbuffers::Offset<FBCudaApiCall>
gpuless::CudnnCreate::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCudnnCreate(builder, this->virtual_handle);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudnnCreate, api_call.Union());
    return api_call_union;
}

CudnnCreate::CudnnCreate(const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCudnnCreate();
    this->virtual_handle = c->virtual_handle();
}

/*
 * cudnnSetStream
 */
uint64_t gpuless::CudnnSetStream::executeNative(CudaVirtualDevice &vdev) {
    static auto real =
        (decltype(&cudnnSetStream))real_dlsym(RTLD_NEXT, "cudnnSetStream");
    return real(vdev.cudnn_handles_virtual_to_real[this->virtual_handle],
                this->stream);
}

gpuless::CudnnSetStream::CudnnSetStream(uint64_t virtualHandle,
                                        cudaStream_t stream)
    : virtual_handle(virtualHandle), stream(stream) {}

flatbuffers::Offset<FBCudaApiCall>
CudnnSetStream::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {
    auto api_call =
        CreateFBCudnnSetStream(builder, this->virtual_handle,
                               reinterpret_cast<uint64_t>(this->stream));
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudnnSetStream, api_call.Union());
    return api_call_union;
}

CudnnSetStream::CudnnSetStream(const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCudnnSetStream();
    this->virtual_handle = c->virtual_handle();
    this->stream = reinterpret_cast<cudaStream_t>(c->stream());
}

/*
 * cudnnCreateTensorDescriptor
 */
uint64_t
gpuless::CudnnCreateTensorDescriptor::executeNative(CudaVirtualDevice &vdev) {
    static auto real = (decltype(&cudnnCreateTensorDescriptor))real_dlsym(
        RTLD_NEXT, "cudnnCreateTensorDescriptor");
    if (vdev.cudnn_tensor_descriptor_virtual_to_real.size() <
        this->virtual_td + 1) {
        vdev.cudnn_tensor_descriptor_virtual_to_real.resize(this->virtual_td +
                                                            1);
    }
    return real(
        &vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td]);
}

gpuless::CudnnCreateTensorDescriptor::CudnnCreateTensorDescriptor(
    uint64_t virtualTd)
    : virtual_td(virtualTd) {}

flatbuffers::Offset<FBCudaApiCall> CudnnCreateTensorDescriptor::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    auto api_call =
        CreateFBCudnnCreateTensorDescriptor(builder, this->virtual_td);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudnnCreateTensorDescriptor,
        api_call.Union());
    return api_call_union;
}

CudnnCreateTensorDescriptor::CudnnCreateTensorDescriptor(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCudnnCreateTensorDescriptor();
    this->virtual_td = c->virtual_td();
}

/*
 * cudnnSetTensorNdDescriptor
 */
uint64_t
gpuless::CudnnSetTensorNdDescriptor::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cudnnSetTensorNdDescriptor);

    cudnnTensorDescriptor_t td =
        vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td];

    return real(td, this->data_type, this->nb_dims, this->dim_a.data(),
                this->stride_a.data());
}

gpuless::CudnnSetTensorNdDescriptor::CudnnSetTensorNdDescriptor(
    uint64_t virtualTd, cudnnDataType_t dataType, int nbDims,
    std::vector<int> dimA, std::vector<int> strideA)
    : virtual_td(virtualTd), data_type(dataType), nb_dims(nbDims),
      dim_a(std::move(dimA)), stride_a(std::move(strideA)) {}

flatbuffers::Offset<FBCudaApiCall> CudnnSetTensorNdDescriptor::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCudnnSetTensorNdDescriptor(
        builder, this->virtual_td, this->data_type, this->nb_dims,
        builder.CreateVector(this->dim_a),
        builder.CreateVector(this->stride_a));
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudnnSetTensorNdDescriptor,
        api_call.Union());
    return api_call_union;
}

CudnnSetTensorNdDescriptor::CudnnSetTensorNdDescriptor(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCudnnSetTensorNdDescriptor();
    this->virtual_td = c->virtual_td();
    this->data_type = static_cast<cudnnDataType_t>(c->data_type());
    this->nb_dims = c->nb_dims();
    this->dim_a = std::vector<int>(c->dim_a()->begin(), c->dim_a()->end());
    this->stride_a =
        std::vector<int>(c->stride_a()->begin(), c->stride_a()->end());
}

/*
 * cudnnCreateFilterDescriptor
 */
uint64_t
gpuless::CudnnCreateFilterDescriptor::executeNative(CudaVirtualDevice &vdev) {
    static auto real = (decltype(&cudnnCreateFilterDescriptor))real_dlsym(
        RTLD_NEXT, "cudnnCreateFilterDescriptor");
    if (vdev.cudnn_filter_descriptor_virtual_to_real.size() <
        this->virtual_fd + 1) {
        vdev.cudnn_filter_descriptor_virtual_to_real.resize(this->virtual_fd +
                                                            1);
    }
    return real(
        &vdev.cudnn_filter_descriptor_virtual_to_real[this->virtual_fd]);
}

gpuless::CudnnCreateFilterDescriptor::CudnnCreateFilterDescriptor(
    uint64_t virtualFd)
    : virtual_fd(virtualFd) {}

flatbuffers::Offset<FBCudaApiCall> CudnnCreateFilterDescriptor::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    auto api_call =
        CreateFBCudnnCreateFilterDescriptor(builder, this->virtual_fd);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudnnCreateFilterDescriptor,
        api_call.Union());
    return api_call_union;
}

CudnnCreateFilterDescriptor::CudnnCreateFilterDescriptor(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCudnnCreateFilterDescriptor();
    this->virtual_fd = c->virtual_fd();
}

/*
 * cudnnSetFilterNdDescriptor
 */
uint64_t
gpuless::CudnnSetFilterNdDescriptor::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cudnnSetFilterNdDescriptor);

    cudnnFilterDescriptor_t fd =
        vdev.cudnn_filter_descriptor_virtual_to_real[this->virtual_fd];

    return real(fd, this->data_type, this->format, this->nb_dims,
                this->filter_dim_a.data());
}

gpuless::CudnnSetFilterNdDescriptor::CudnnSetFilterNdDescriptor(
    uint64_t virtualFd, cudnnDataType_t dataType, cudnnTensorFormat_t format,
    int nbDims, const std::vector<int> &filterDimA)
    : virtual_fd(virtualFd), data_type(dataType), format(format),
      nb_dims(nbDims), filter_dim_a(filterDimA) {}

flatbuffers::Offset<FBCudaApiCall> CudnnSetFilterNdDescriptor::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCudnnSetFilterNdDescriptor(
        builder, this->virtual_fd, this->data_type, this->format, this->nb_dims,
        builder.CreateVector(this->filter_dim_a));
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudnnSetFilterNdDescriptor,
        api_call.Union());
    return api_call_union;
}

CudnnSetFilterNdDescriptor::CudnnSetFilterNdDescriptor(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCudnnSetFilterNdDescriptor();
    this->virtual_fd = c->virtual_fd();
    this->data_type = static_cast<cudnnDataType_t>(c->data_type());
    this->format = static_cast<cudnnTensorFormat_t>(c->format());
    this->nb_dims = c->nb_dims();
    this->filter_dim_a =
        std::vector<int>(c->filter_dim_a()->begin(), c->filter_dim_a()->end());
}

/*
 * cudnnCreateConvolutionDescriptor
 */
uint64_t gpuless::CudnnCreateConvolutionDescriptor::executeNative(
    CudaVirtualDevice &vdev) {
    static auto real = (decltype(&cudnnCreateConvolutionDescriptor))real_dlsym(
        RTLD_NEXT, "cudnnCreateConvolutionDescriptor");
    if (vdev.cudnn_convolution_descriptor_virtual_to_real.size() <
        this->virtual_cd + 1) {
        vdev.cudnn_convolution_descriptor_virtual_to_real.resize(
            this->virtual_cd + 1);
    }
    return real(
        &vdev.cudnn_convolution_descriptor_virtual_to_real[this->virtual_cd]);
}

gpuless::CudnnCreateConvolutionDescriptor::CudnnCreateConvolutionDescriptor(
    uint64_t virtualCd)
    : virtual_cd(virtualCd) {}

flatbuffers::Offset<FBCudaApiCall>
CudnnCreateConvolutionDescriptor::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    auto api_call =
        CreateFBCudnnCreateConvolutionDescriptor(builder, this->virtual_cd);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudnnCreateConvolutionDescriptor,
        api_call.Union());
    return api_call_union;
}

CudnnCreateConvolutionDescriptor::CudnnCreateConvolutionDescriptor(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCudnnCreateConvolutionDescriptor();
    this->virtual_cd = c->virtual_cd();
}

/*
 * cudnnSetConvolutionGroupCount
 */
uint64_t
gpuless::CudnnSetConvolutionGroupCount::executeNative(CudaVirtualDevice &vdev) {
    static auto real = (decltype(&cudnnSetConvolutionGroupCount))real_dlsym(
        RTLD_NEXT, "cudnnSetConvolutionGroupCount");
    return real(
        vdev.cudnn_convolution_descriptor_virtual_to_real[this->virtual_cd],
        this->group_count);
}

gpuless::CudnnSetConvolutionGroupCount::CudnnSetConvolutionGroupCount(
    uint64_t virtualCd, int groupCount)
    : virtual_cd(virtualCd), group_count(groupCount) {}

flatbuffers::Offset<FBCudaApiCall> CudnnSetConvolutionGroupCount::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCudnnSetConvolutionGroupCount(
        builder, this->virtual_cd, this->group_count);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudnnSetConvolutionGroupCount,
        api_call.Union());
    return api_call_union;
}

CudnnSetConvolutionGroupCount::CudnnSetConvolutionGroupCount(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCudnnSetConvolutionGroupCount();
    this->virtual_cd = c->virtual_cd();
    this->group_count = c->group_count();
}

/*
 * cudnnSetConvolutionMathType
 */
uint64_t
gpuless::CudnnSetConvolutionMathType::executeNative(CudaVirtualDevice &vdev) {
    static auto real = (decltype(&cudnnSetConvolutionMathType))real_dlsym(
        RTLD_NEXT, "cudnnSetConvolutionMathType");
    return real(
        vdev.cudnn_convolution_descriptor_virtual_to_real[this->virtual_cd],
        this->math_type);
}

gpuless::CudnnSetConvolutionMathType::CudnnSetConvolutionMathType(
    uint64_t virtualCd, cudnnMathType_t mathType)
    : virtual_cd(virtualCd), math_type(mathType) {}

flatbuffers::Offset<FBCudaApiCall> CudnnSetConvolutionMathType::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCudnnSetConvolutionMathType(
        builder, this->virtual_cd, this->math_type);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudnnSetConvolutionMathType,
        api_call.Union());
    return api_call_union;
}

CudnnSetConvolutionMathType::CudnnSetConvolutionMathType(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCudnnSetConvolutionMathType();
    this->virtual_cd = c->virtual_cd();
    this->math_type = static_cast<cudnnMathType_t>(c->math_type());
}

/*
 * cudnnSetConvolutionNdDescriptor
 */
uint64_t gpuless::CudnnSetConvolutionNdDescriptor::executeNative(
    CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cudnnSetConvolutionNdDescriptor);

    cudnnConvolutionDescriptor_t cd =
        vdev.cudnn_convolution_descriptor_virtual_to_real[this->virtual_cd];

    return real(cd, this->array_length, this->pad_a.data(),
                this->filter_stride_a.data(), this->dilation.data(),
                this->convolution_mode, this->cudnn_data_type);
}

gpuless::CudnnSetConvolutionNdDescriptor::CudnnSetConvolutionNdDescriptor(
    uint64_t virtualCd, int arrayLength, std::vector<int> padA,
    std::vector<int> filterStrideA, std::vector<int> dilation,
    cudnnConvolutionMode_t convolutionMode, cudnnDataType_t cudnnDataType)
    : virtual_cd(virtualCd), array_length(arrayLength), pad_a(std::move(padA)),
      filter_stride_a(std::move(filterStrideA)), dilation(std::move(dilation)),
      convolution_mode(convolutionMode), cudnn_data_type(cudnnDataType) {}

flatbuffers::Offset<FBCudaApiCall> CudnnSetConvolutionNdDescriptor::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCudnnSetConvolutionNdDescriptor(
        builder, this->virtual_cd, this->array_length,
        builder.CreateVector(this->pad_a),
        builder.CreateVector(this->filter_stride_a),
        builder.CreateVector(this->dilation), this->convolution_mode,
        this->cudnn_data_type);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudnnSetConvolutionNdDescriptor,
        api_call.Union());
    return api_call_union;
}

CudnnSetConvolutionNdDescriptor::CudnnSetConvolutionNdDescriptor(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCudnnSetConvolutionNdDescriptor();
    this->virtual_cd = c->virtual_cd();
    this->array_length = c->array_length();
    this->pad_a = std::vector<int>(c->pad_a()->begin(), c->pad_a()->end());
    this->filter_stride_a = std::vector<int>(c->filter_stride_a()->begin(),
                                             c->filter_stride_a()->end());
    this->dilation =
        std::vector<int>(c->dilation()->begin(), c->dilation()->end());
    this->convolution_mode =
        static_cast<cudnnConvolutionMode_t>(c->convolution_mode());
    this->cudnn_data_type = static_cast<cudnnDataType_t>(c->cudnn_data_type());
}

/*
 * cudnnGetConvolutionForwardAlgorithm_v7
 */
uint64_t gpuless::CudnnGetConvolutionForwardAlgorithmV7::executeNative(
    CudaVirtualDevice &vdev) {
    static auto real =
        GET_REAL_FUNCTION(cudnnGetConvolutionForwardAlgorithm_v7);

    cudnnHandle_t handle =
        vdev.cudnn_handles_virtual_to_real[this->virtual_handle];
    cudnnTensorDescriptor_t td_xdexc =
        vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td_xdesc];
    cudnnFilterDescriptor_t fd =
        vdev.cudnn_filter_descriptor_virtual_to_real[this->virtual_fd];
    cudnnConvolutionDescriptor_t cd =
        vdev.cudnn_convolution_descriptor_virtual_to_real[this->virtual_cd];
    cudnnTensorDescriptor_t td_ydesc =
        vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td_ydesc];

    return real(handle, td_xdexc, fd, cd, td_ydesc, this->requested_algo_count,
                &this->returned_algo_count, this->perf_results.data());
}

gpuless::CudnnGetConvolutionForwardAlgorithmV7::
    CudnnGetConvolutionForwardAlgorithmV7(uint64_t virtualHandle,
                                          uint64_t virtualTdXdesc,
                                          uint64_t virtualTdYdesc,
                                          uint64_t virtualFd,
                                          uint64_t virtualCd,
                                          int requestedAlgoCount)
    : virtual_handle(virtualHandle), virtual_td_xdesc(virtualTdXdesc),
      virtual_td_ydesc(virtualTdYdesc), virtual_fd(virtualFd),
      virtual_cd(virtualCd), requested_algo_count(requestedAlgoCount),
      perf_results(requestedAlgoCount) {}

flatbuffers::Offset<FBCudaApiCall>
CudnnGetConvolutionForwardAlgorithmV7::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    std::vector<flatbuffers::Offset<FBCudnnConvolutionFwdAlgoPerf>>
        perf_results_vec;
    for (const auto &p : this->perf_results) {
        std::vector<int> reserved_vec(3);
        reserved_vec[0] = p.reserved[0];
        reserved_vec[1] = p.reserved[1];
        reserved_vec[2] = p.reserved[2];
        perf_results_vec.push_back(CreateFBCudnnConvolutionFwdAlgoPerf(
            builder, p.algo, p.status, p.time, p.memory, p.determinism,
            p.mathType, builder.CreateVector(reserved_vec)));
    }
    auto api_call = CreateFBCudnnGetConvolutionForwardAlgorithmV7(
        builder, this->virtual_handle, this->virtual_td_xdesc,
        this->virtual_td_ydesc, this->virtual_fd, this->virtual_cd,
        this->requested_algo_count, this->returned_algo_count,
        builder.CreateVector(perf_results_vec));
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudnnGetConvolutionForwardAlgorithmV7,
        api_call.Union());
    return api_call_union;
}

CudnnGetConvolutionForwardAlgorithmV7::CudnnGetConvolutionForwardAlgorithmV7(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c =
        fb_cuda_api_call->api_call_as_FBCudnnGetConvolutionForwardAlgorithmV7();
    this->virtual_handle = c->virtual_handle();
    this->virtual_td_xdesc = c->virtual_td_xdesc();
    this->virtual_td_ydesc = c->virtual_td_ydesc();
    this->virtual_fd = c->virtual_fd();
    this->virtual_cd = c->virtual_cd();
    this->requested_algo_count = c->requested_algo_count();
    this->returned_algo_count = c->returned_algo_count();
    this->perf_results = std::vector<cudnnConvolutionFwdAlgoPerf_t>();
    for (const auto &p : *c->perf_results()) {
        auto perf = cudnnConvolutionFwdAlgoPerf_t{};
        perf.algo = static_cast<cudnnConvolutionFwdAlgo_t>(p->algo());
        perf.status = static_cast<cudnnStatus_t>(p->status());
        perf.time = p->time();
        perf.memory = p->memory();
        perf.determinism = static_cast<cudnnDeterminism_t>(p->determinism());
        perf.mathType = static_cast<cudnnMathType_t>(p->math_type());
        perf.reserved[0] = p->reserved()->Get(0);
        perf.reserved[1] = p->reserved()->Get(1);
        perf.reserved[2] = p->reserved()->Get(2);
        this->perf_results.push_back(perf);
    }
    this->perf_results.resize(requested_algo_count);
}

/*
 * cudnnConvolutionForward
 */
uint64_t
gpuless::CudnnConvolutionForward::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cudnnConvolutionForward);
    cudnnHandle_t handle =
        vdev.cudnn_handles_virtual_to_real[this->virtual_handle];
    const cudnnTensorDescriptor_t xDesc =
        vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td_xdesc];
    const cudnnFilterDescriptor_t wDesc =
        vdev.cudnn_filter_descriptor_virtual_to_real[this->virtual_fd_wdesc];
    cudnnConvolutionDescriptor_t convDesc =
        vdev.cudnn_convolution_descriptor_virtual_to_real[this->virtual_cd];
    const cudnnTensorDescriptor_t yDesc =
        vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td_ydesc];

    return real(handle, this->alpha.data(), xDesc, this->x, wDesc, this->w,
                convDesc, this->algo, this->workspace,
                this->workspace_size_in_bytes, this->beta.data(), yDesc,
                this->y);
}

gpuless::CudnnConvolutionForward::CudnnConvolutionForward(
    uint64_t virtualHandle, size_t scaling_size, const void *alpha_ptr,
    const void *beta_ptr, void *workspace, size_t workspaceSizeInBytes,
    uint64_t virtualCd, cudnnConvolutionFwdAlgo_t algo, uint64_t virtualFdWdesc,
    const void *w, uint64_t virtualTdXdesc, const void *x,
    uint64_t virtualTdYdesc, void *y)
    : virtual_handle(virtualHandle), alpha(scaling_size), beta(scaling_size),
      workspace(workspace), workspace_size_in_bytes(workspaceSizeInBytes),
      virtual_cd(virtualCd), algo(algo), virtual_fd_wdesc(virtualFdWdesc), w(w),
      virtual_td_xdesc(virtualTdXdesc), x(x), virtual_td_ydesc(virtualTdYdesc),
      y(y) {
    std::memcpy(this->alpha.data(), alpha_ptr, scaling_size);
    std::memcpy(this->beta.data(), beta_ptr, scaling_size);
}

flatbuffers::Offset<FBCudaApiCall>
CudnnConvolutionForward::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCudnnConvolutionForward(
        builder, this->virtual_handle, builder.CreateVector(this->alpha),
        builder.CreateVector(this->beta),
        reinterpret_cast<uint64_t>(this->workspace),
        this->workspace_size_in_bytes, this->virtual_cd, this->algo,
        this->virtual_fd_wdesc, reinterpret_cast<uint64_t>(this->w),
        this->virtual_td_xdesc, reinterpret_cast<uint64_t>(this->x),
        this->virtual_td_ydesc, reinterpret_cast<uint64_t>(this->y));
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudnnConvolutionForward,
        api_call.Union());
    return api_call_union;
}

CudnnConvolutionForward::CudnnConvolutionForward(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCudnnConvolutionForward();
    this->virtual_handle = c->virtual_handle();
    this->alpha = std::vector<uint8_t>(c->alpha()->begin(), c->alpha()->end());
    this->beta = std::vector<uint8_t>(c->beta()->begin(), c->beta()->end());
    this->workspace = reinterpret_cast<void *>(c->workspace());
    this->workspace_size_in_bytes = c->workspace_size_in_bytes();
    this->virtual_cd = c->virtual_cd();
    this->algo = static_cast<cudnnConvolutionFwdAlgo_t>(c->algo());
    this->virtual_fd_wdesc = c->virtual_fd_wdesc();
    this->w = reinterpret_cast<void *>(c->w());
    this->virtual_td_xdesc = c->virtual_td_xdesc();
    this->x = reinterpret_cast<const void *>(c->x());
    this->virtual_td_ydesc = c->virtual_td_ydesc();
    this->y = reinterpret_cast<void *>(c->y());
}

/*
 * cudnnBatchNormalizationForwardInference
 */
uint64_t gpuless::CudnnBatchNormalizationForwardInference::executeNative(
    CudaVirtualDevice &vdev) {
    static auto real =
        GET_REAL_FUNCTION(cudnnBatchNormalizationForwardInference);

    cudnnHandle_t handle =
        vdev.cudnn_handles_virtual_to_real[this->virtual_handle];
    const cudnnTensorDescriptor_t xDesc =
        vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td_xdesc];
    const cudnnTensorDescriptor_t yDesc =
        vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td_ydesc];
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc =
        vdev.cudnn_tensor_descriptor_virtual_to_real
            [this->virtual_td_bs_scale_bias_mean_var_desc];

    return real(handle, this->mode, this->alpha.data(), this->beta.data(),
                xDesc, this->x, yDesc, this->y, bnScaleBiasMeanVarDesc,
                this->bn_scale, this->bn_bias, this->estimated_mean,
                this->estimated_variance, this->epsilon);
}

gpuless::CudnnBatchNormalizationForwardInference::
    CudnnBatchNormalizationForwardInference(
        uint64_t virtualHandle, cudnnBatchNormMode_t mode, size_t scaling_size,
        const void *alpha_ptr, const void *beta_ptr, uint64_t virtualTdXdesc,
        const void *x, uint64_t virtualTdYdesc, void *y,
        uint64_t virtualTdBsScaleBiasMeanVarDesc, const void *bnScale,
        const void *bnBias, const void *estimatedMean,
        const void *estimatedVariance, double epsilon)
    : virtual_handle(virtualHandle), mode(mode), alpha(scaling_size),
      beta(scaling_size), virtual_td_xdesc(virtualTdXdesc), x(x),
      virtual_td_ydesc(virtualTdYdesc), y(y),
      virtual_td_bs_scale_bias_mean_var_desc(virtualTdBsScaleBiasMeanVarDesc),
      bn_scale(bnScale), bn_bias(bnBias), estimated_mean(estimatedMean),
      estimated_variance(estimatedVariance), epsilon(epsilon) {
    std::memcpy(this->alpha.data(), alpha_ptr, scaling_size);
    std::memcpy(this->beta.data(), beta_ptr, scaling_size);
}

flatbuffers::Offset<FBCudaApiCall>
CudnnBatchNormalizationForwardInference::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCudnnBatchNormalizationForwardInference(
        builder, this->virtual_handle, this->mode,
        builder.CreateVector(this->alpha), builder.CreateVector(this->beta),
        this->virtual_td_xdesc, reinterpret_cast<uint64_t>(this->x),
        this->virtual_td_ydesc, reinterpret_cast<uint64_t>(this->y),
        this->virtual_td_bs_scale_bias_mean_var_desc,
        reinterpret_cast<uint64_t>(this->bn_scale),
        reinterpret_cast<uint64_t>(this->bn_bias),
        reinterpret_cast<uint64_t>(this->estimated_mean),
        reinterpret_cast<uint64_t>(this->estimated_variance), this->epsilon);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudnnBatchNormalizationForwardInference,
        api_call.Union());
    return api_call_union;
}

CudnnBatchNormalizationForwardInference::
    CudnnBatchNormalizationForwardInference(
        const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call
                 ->api_call_as_FBCudnnBatchNormalizationForwardInference();
    this->virtual_handle = c->virtual_handle();
    this->mode = static_cast<cudnnBatchNormMode_t>(c->mode());
    this->alpha = std::vector<uint8_t>(c->alpha()->begin(), c->alpha()->end());
    this->beta = std::vector<uint8_t>(c->beta()->begin(), c->beta()->end());
    this->virtual_td_xdesc = c->virtual_td_xdesc();
    this->x = reinterpret_cast<const void *>(c->x());
    this->virtual_td_ydesc = c->virtual_td_ydesc();
    this->y = reinterpret_cast<void *>(c->y());
    this->virtual_td_bs_scale_bias_mean_var_desc =
        c->virtual_td_bs_scale_bias_mean_var_desc();
    this->bn_scale = reinterpret_cast<void *>(c->bn_scale());
    this->bn_bias = reinterpret_cast<void *>(c->bn_bias());
    this->estimated_mean = reinterpret_cast<void *>(c->estimated_mean());
    this->estimated_variance =
        reinterpret_cast<void *>(c->estimated_variance());
    this->epsilon = c->epsilon();
}

/*
 * cudnnDestroyConvolutionDescriptor
 */
gpuless::CudnnDestroyConvolutionDescriptor::CudnnDestroyConvolutionDescriptor(
    uint64_t virtualCd)
    : virtual_cd(virtualCd) {}

uint64_t gpuless::CudnnDestroyConvolutionDescriptor::executeNative(
    CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cudnnDestroyConvolutionDescriptor);
    return real(
        vdev.cudnn_convolution_descriptor_virtual_to_real[this->virtual_cd]);
}

flatbuffers::Offset<FBCudaApiCall>
CudnnDestroyConvolutionDescriptor::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    auto api_call =
        CreateFBCudnnDestroyConvolutionDescriptor(builder, this->virtual_cd);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudnnDestroyConvolutionDescriptor,
        api_call.Union());
    return api_call_union;
}

CudnnDestroyConvolutionDescriptor::CudnnDestroyConvolutionDescriptor(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c =
        fb_cuda_api_call->api_call_as_FBCudnnDestroyConvolutionDescriptor();
    this->virtual_cd = c->virtual_cd();
}

/*
 * cudnnDestroyFilterDescriptor
 */
gpuless::CudnnDestroyFilterDescriptor::CudnnDestroyFilterDescriptor(
    uint64_t virtualFd)
    : virtual_fd(virtualFd) {}

uint64_t
gpuless::CudnnDestroyFilterDescriptor::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cudnnDestroyFilterDescriptor);
    return real(vdev.cudnn_filter_descriptor_virtual_to_real[this->virtual_fd]);
}

flatbuffers::Offset<FBCudaApiCall> CudnnDestroyFilterDescriptor::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    auto api_call =
        CreateFBCudnnDestroyFilterDescriptor(builder, this->virtual_fd);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudnnDestroyFilterDescriptor,
        api_call.Union());
    return api_call_union;
}

CudnnDestroyFilterDescriptor::CudnnDestroyFilterDescriptor(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCudnnDestroyFilterDescriptor();
    this->virtual_fd = c->virtual_fd();
}

/*
 * cudnnDestroyTensorDescriptor
 */
gpuless::CudnnDestroyTensorDescriptor::CudnnDestroyTensorDescriptor(
    uint64_t virtualTd)
    : virtual_td(virtualTd) {}

uint64_t
gpuless::CudnnDestroyTensorDescriptor::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cudnnDestroyTensorDescriptor);
    return real(vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td]);
}

flatbuffers::Offset<FBCudaApiCall> CudnnDestroyTensorDescriptor::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    auto api_call =
        CreateFBCudnnDestroyTensorDescriptor(builder, this->virtual_td);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudnnDestroyTensorDescriptor,
        api_call.Union());
    return api_call_union;
}

CudnnDestroyTensorDescriptor::CudnnDestroyTensorDescriptor(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCudnnDestroyTensorDescriptor();
    this->virtual_td = c->virtual_td();
}

/*
 * cudnnConvolutionBackwardData
 */
CudnnConvolutionBackwardData::CudnnConvolutionBackwardData(
    const uint64_t &virtualHandle, const std::vector<uint8_t> &alpha,
    const uint64_t &virtualFdWdesc, const void *w,
    const uint64_t &virtualTdDydesc, const void *dy, const uint64_t &virtualCd,
    cudnnConvolutionBwdDataAlgo_t algo, void *workspace,
    size_t workspaceSizeInBytes, const std::vector<uint8_t> &beta,
    const uint64_t &virtualTdDxdesc, void *dx)
    : virtual_handle(virtualHandle), alpha(alpha),
      virtual_fd_wdesc(virtualFdWdesc), w(w),
      virtual_td_dydesc(virtualTdDydesc), dy(dy), virtual_cd(virtualCd),
      algo(algo), workspace(workspace),
      workspace_size_in_bytes(workspaceSizeInBytes), beta(beta),
      virtual_td_dxdesc(virtualTdDxdesc), dx(dx) {}

CudnnConvolutionBackwardData::CudnnConvolutionBackwardData(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCudnnConvolutionBackwardData();
    this->virtual_handle = c->virtual_handle();
    this->alpha.resize(c->alpha()->size());
    this->alpha.insert(this->alpha.begin(), c->alpha()->begin(),
                       c->alpha()->end());
    this->beta.resize(c->beta()->size());
    this->beta.insert(this->beta.begin(), c->beta()->begin(), c->beta()->end());
    this->virtual_fd_wdesc = c->virtual_fd_wdesc();
    this->w = reinterpret_cast<const void *>(c->w());
    this->virtual_td_dydesc = c->virtual_td_dydesc();
    this->dy = reinterpret_cast<const void *>(c->dy());
    this->virtual_cd = c->virtual_cd();
    this->algo = static_cast<cudnnConvolutionBwdDataAlgo_t>(c->algo());
    this->workspace = reinterpret_cast<void *>(c->workspace());
    this->workspace_size_in_bytes = c->workspace_size_in_bytes();
    this->virtual_td_dxdesc = c->virtual_td_dxdesc();
    this->dx = reinterpret_cast<void *>(c->dx());
}

uint64_t CudnnConvolutionBackwardData::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cudnnConvolutionBackwardData);
    return real(
        vdev.cudnn_handles_virtual_to_real[this->virtual_handle],
        this->alpha.data(),
        vdev.cudnn_filter_descriptor_virtual_to_real[this->virtual_fd_wdesc],
        this->w,
        vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td_dydesc],
        dy, vdev.cudnn_convolution_descriptor_virtual_to_real[this->virtual_cd],
        this->algo, this->workspace, this->workspace_size_in_bytes,
        this->beta.data(),
        vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td_dxdesc],
        this->dx

    );
}

flatbuffers::Offset<FBCudaApiCall> CudnnConvolutionBackwardData::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCudnnConvolutionBackwardData(
        builder, this->virtual_handle, builder.CreateVector(this->alpha),
        this->virtual_fd_wdesc, reinterpret_cast<uint64_t>(this->w),
        this->virtual_td_dydesc, reinterpret_cast<uint64_t>(this->dy),
        this->virtual_cd, this->algo,
        reinterpret_cast<uint64_t>(this->workspace),
        this->workspace_size_in_bytes, builder.CreateVector(beta),
        this->virtual_td_dxdesc, reinterpret_cast<uint64_t>(this->dx));
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudnnConvolutionBackwardData,
        api_call.Union());
    return api_call_union;
}

/*
 * cudnnGetConvolutionBackwardDataAlgorithmV7
 */
CudnnGetConvolutionBackwardDataAlgorithmV7::
    CudnnGetConvolutionBackwardDataAlgorithmV7(uint64_t virtualHandle,
                                               uint64_t virtualFdWdesc,
                                               uint64_t virtualTdDydesc,
                                               uint64_t virtualCdConvdesc,
                                               uint64_t virtualTdDxdesc,
                                               int requestedAlgoCount)
    : virtual_handle(virtualHandle), virtual_fd_wdesc(virtualFdWdesc),
      virtual_td_dydesc(virtualTdDydesc),
      virtual_cd_convdesc(virtualCdConvdesc),
      virtual_td_dxdesc(virtualTdDxdesc),
      requested_algo_count(requestedAlgoCount),
      perf_results(requestedAlgoCount) {}

CudnnGetConvolutionBackwardDataAlgorithmV7::
    CudnnGetConvolutionBackwardDataAlgorithmV7(
        const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call
                 ->api_call_as_FBCudnnGetConvolutionBackwardDataAlgorithmV7();
    this->virtual_handle = c->virtual_handle();
    this->virtual_fd_wdesc = c->virtual_fd_wdesc();
    this->virtual_td_dydesc = c->virtual_td_dydesc();
    this->virtual_cd_convdesc = c->virtual_cd_convdesc();
    this->virtual_td_dxdesc = c->virtual_td_dxdesc();
    this->requested_algo_count = c->requested_algo_count();
    this->returned_algo_count = c->returned_algo_count();

    std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results_;
    for (const auto &p : *c->perf_results()) {
        cudnnConvolutionBwdDataAlgoPerf_t perf;
        perf.algo = static_cast<cudnnConvolutionBwdDataAlgo_t>(p->algo());
        perf.status = static_cast<cudnnStatus_t>(p->status());
        perf.time = p->time();
        perf.memory = p->memory();
        perf.determinism = static_cast<cudnnDeterminism_t>(p->determinism());
        perf.mathType = static_cast<cudnnMathType_t>(p->math_type());
        perf.reserved[0] = p->reserved()->Get(0);
        perf.reserved[1] = p->reserved()->Get(1);
        perf.reserved[2] = p->reserved()->Get(2);
        perf_results_.push_back(perf);
    }

    this->perf_results = perf_results_;
}

uint64_t CudnnGetConvolutionBackwardDataAlgorithmV7::executeNative(
    CudaVirtualDevice &vdev) {
    static auto real =
        GET_REAL_FUNCTION(cudnnGetConvolutionBackwardDataAlgorithm_v7);

    cudnnHandle_t handle =
        vdev.cudnn_handles_virtual_to_real[this->virtual_handle];
    cudnnFilterDescriptor_t wdesc =
        vdev.cudnn_filter_descriptor_virtual_to_real[this->virtual_fd_wdesc];
    cudnnTensorDescriptor_t dydesc =
        vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td_dydesc];
    cudnnConvolutionDescriptor_t convdesc =
        vdev.cudnn_convolution_descriptor_virtual_to_real
            [this->virtual_cd_convdesc];
    cudnnTensorDescriptor_t dxdesc =
        vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td_dxdesc];

    return real(handle, wdesc, dydesc, convdesc, dxdesc,
                  this->requested_algo_count, &this->returned_algo_count,
                  this->perf_results.data());
}

flatbuffers::Offset<FBCudaApiCall>
CudnnGetConvolutionBackwardDataAlgorithmV7::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    std::vector<flatbuffers::Offset<FBCudnnConvolutionBwdDataAlgoPerf>>
        perf_results_vec;
    for (const auto &p : this->perf_results) {
        std::vector<int> reserved_(3);
        reserved_[0] = p.reserved[0];
        reserved_[1] = p.reserved[1];
        reserved_[2] = p.reserved[2];
        auto perf = CreateFBCudnnConvolutionBwdDataAlgoPerf(
            builder, p.algo, p.status, p.time, p.memory, p.determinism,
            p.mathType, builder.CreateVector(reserved_));
        perf_results_vec.push_back(perf);
    }

    auto api_call = CreateFBCudnnGetConvolutionBackwardDataAlgorithmV7(
        builder, this->virtual_handle, this->virtual_fd_wdesc,
        this->virtual_td_dydesc, this->virtual_cd_convdesc,
        this->virtual_td_dxdesc, this->requested_algo_count,
        this->returned_algo_count, builder.CreateVector(perf_results_vec));
    auto api_call_union = CreateFBCudaApiCall(
        builder,
        FBCudaApiCallUnion_FBCudnnGetConvolutionBackwardDataAlgorithmV7,
        api_call.Union());
    return api_call_union;
}

/*
 * cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize
 */
CudnnGetBatchNormalizationForwardTrainingExWorkspaceSize::
    CudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
        uint64_t virtualHandle, cudnnBatchNormMode_t mode,
        cudnnBatchNormOps_t bnOps, uint64_t virtualTdXdesc,
        uint64_t virtualTdZdesc, uint64_t virtualTdYdesc,
        uint64_t virtualTdBnScaleBiasMeanVarDesc,
        uint64_t virtualAdActivationDesc)
    : virtual_handle(virtualHandle), mode(mode), bn_ops(bnOps),
      virtual_td_xdesc(virtualTdXdesc), virtual_td_zdesc(virtualTdZdesc),
      virtual_td_ydesc(virtualTdYdesc),
      virtual_td_bn_scale_bias_mean_var_desc(virtualTdBnScaleBiasMeanVarDesc),
      virtual_ad_activation_desc(virtualAdActivationDesc) {}

CudnnGetBatchNormalizationForwardTrainingExWorkspaceSize::
    CudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
        const FBCudaApiCall *fb_cuda_api_call) {
    auto c =
        fb_cuda_api_call
            ->api_call_as_FBCudnnGetBatchNormalizationForwardTrainingExWorkspaceSize();
    this->virtual_handle = c->virtual_handle();
    this->mode = static_cast<cudnnBatchNormMode_t>(c->mode());
    this->bn_ops = static_cast<cudnnBatchNormOps_t>(c->bn_ops());
    this->virtual_td_xdesc = c->virtual_td_xdesc();
    this->virtual_td_zdesc = c->virtual_td_zdesc();
    this->virtual_td_ydesc = c->virtual_td_ydesc();
    this->virtual_td_bn_scale_bias_mean_var_desc =
        c->virtual_td_bn_scale_bias_mean_var_desc();
    this->virtual_ad_activation_desc = c->virtual_ad_activation_desc();
}

uint64_t
CudnnGetBatchNormalizationForwardTrainingExWorkspaceSize::executeNative(
    CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(
        cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize);
    return real(
        vdev.cudnn_handles_virtual_to_real[this->virtual_handle], this->mode,
        this->bn_ops,
        vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td_xdesc],
        vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td_zdesc],
        vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td_ydesc],
        vdev.cudnn_tensor_descriptor_virtual_to_real
            [this->virtual_td_bn_scale_bias_mean_var_desc],
        reinterpret_cast<cudnnActivationDescriptor_t>(
            this->virtual_ad_activation_desc),
        &this->size_in_bytes);
}

flatbuffers::Offset<FBCudaApiCall>
CudnnGetBatchNormalizationForwardTrainingExWorkspaceSize::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    auto api_call =
        CreateFBCudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
            builder, this->virtual_handle, this->mode, this->bn_ops,
            this->virtual_td_xdesc, this->virtual_td_zdesc,
            this->virtual_td_ydesc,
            this->virtual_td_bn_scale_bias_mean_var_desc,
            this->virtual_ad_activation_desc, this->size_in_bytes);
    auto api_call_union = CreateFBCudaApiCall(
        builder,
        FBCudaApiCallUnion_FBCudnnGetBatchNormalizationForwardTrainingExWorkspaceSize,
        api_call.Union());
    return api_call_union;
}

/*
 * cudnnGetBatchNormalizationTrainingExReserveSpaceSize
 */
CudnnGetBatchNormalizationTrainingExReserveSpaceSize::
    CudnnGetBatchNormalizationTrainingExReserveSpaceSize(
        uint64_t virtualHandle, cudnnBatchNormMode_t mode,
        cudnnBatchNormOps_t bnOps, uint64_t virtualAdActivationDesc,
        uint64_t virtualTdXdesc)
    : virtual_handle(virtualHandle), mode(mode), bn_ops(bnOps),
      virtual_ad_activation_desc(virtualAdActivationDesc),
      virtual_td_xdesc(virtualTdXdesc) {}

CudnnGetBatchNormalizationTrainingExReserveSpaceSize::
    CudnnGetBatchNormalizationTrainingExReserveSpaceSize(
        const FBCudaApiCall *fb_cuda_api_call) {
    auto c =
        fb_cuda_api_call
            ->api_call_as_FBCudnnGetBatchNormalizationTrainingExReserveSpaceSize();
    this->virtual_handle = c->virtual_handle();
    this->mode = static_cast<cudnnBatchNormMode_t>(c->mode());
    this->bn_ops = static_cast<cudnnBatchNormOps_t>(c->bn_ops());
    this->virtual_ad_activation_desc = c->virtual_ad_activation_desc();
    this->virtual_td_xdesc = c->virtual_td_xdesc();
    this->size_in_bytes = c->size_in_bytes();
}

uint64_t CudnnGetBatchNormalizationTrainingExReserveSpaceSize::executeNative(
    CudaVirtualDevice &vdev) {
    static auto real =
        GET_REAL_FUNCTION(cudnnGetBatchNormalizationTrainingExReserveSpaceSize);
    return real(
        vdev.cudnn_handles_virtual_to_real[this->virtual_handle], this->mode,
        this->bn_ops,
        reinterpret_cast<cudnnActivationDescriptor_t>(
            this->virtual_ad_activation_desc),
        vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td_xdesc],
        &this->size_in_bytes);
}

flatbuffers::Offset<FBCudaApiCall>
CudnnGetBatchNormalizationTrainingExReserveSpaceSize::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    auto api_call =
        CreateFBCudnnGetBatchNormalizationTrainingExReserveSpaceSize(
            builder, this->virtual_handle, this->mode, this->bn_ops,
            this->virtual_ad_activation_desc, this->virtual_td_xdesc,
            this->size_in_bytes);
    auto api_call_union = CreateFBCudaApiCall(
        builder,
        FBCudaApiCallUnion_FBCudnnGetBatchNormalizationTrainingExReserveSpaceSize,
        api_call.Union());
    return api_call_union;
}

/*
 * cudnnBatchNormalizationForwardTrainingEx
 */
CudnnBatchNormalizationForwardTrainingEx::
    CudnnBatchNormalizationForwardTrainingEx(
        uint64_t virtualHandle, cudnnBatchNormMode_t mode,
        cudnnBatchNormOps_t bnOps, std::vector<uint8_t> &alpha,
        std::vector<uint8_t> &beta, uint64_t virtualTdXdesc, const void *xData,
        uint64_t virtualTdYdesc, void *yData, uint64_t virtualTdZdesc,
        const void *zData, uint64_t virtualTdBnScaleBiasMeanVarDesc,
        const void *bnScaleData, const void *bnBiasData,
        double exponentialAverageFactor, void *resultRunningMeanData,
        void *resultRunningVarianceData, double epsilon, void *saveMean,
        void *saveInvVariance, uint64_t virtualAdActivationDesc,
        void *workspace, size_t workspaceSizeInBytes, void *reserveSpace,
        size_t reserveSpaceSizeInBytes)
    : virtual_handle(virtualHandle), mode(mode), bn_ops(bnOps), alpha(alpha),
      beta(beta), virtual_td_xdesc(virtualTdXdesc), x_data(xData),
      virtual_td_ydesc(virtualTdYdesc), y_data(yData),
      virtual_td_zdesc(virtualTdZdesc), z_data(zData),
      virtual_td_bn_scale_bias_mean_var_desc(virtualTdBnScaleBiasMeanVarDesc),
      bn_scale_data(bnScaleData), bn_bias_data(bnBiasData),
      exponential_average_factor(exponentialAverageFactor),
      result_running_mean_data(resultRunningMeanData),
      result_running_variance_data(resultRunningVarianceData), epsilon(epsilon),
      save_mean(saveMean), save_inv_variance(saveInvVariance),
      virtual_ad_activation_desc(virtualAdActivationDesc), workspace(workspace),
      workspace_size_in_bytes(workspaceSizeInBytes),
      reserve_space(reserveSpace),
      reserve_space_size_in_bytes(reserveSpaceSizeInBytes) {}

CudnnBatchNormalizationForwardTrainingEx::
    CudnnBatchNormalizationForwardTrainingEx(
        const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call
                 ->api_call_as_FBCudnnBatchNormalizationForwardTrainingEx();
    this->virtual_handle = c->virtual_handle();
    this->mode = static_cast<cudnnBatchNormMode_t>(c->mode());
    this->bn_ops = static_cast<cudnnBatchNormOps_t>(c->bn_ops());
    this->alpha.resize(c->alpha()->size());
    this->alpha.insert(this->alpha.begin(), c->alpha()->begin(),
                       c->alpha()->end());
    this->beta.resize(c->beta()->size());
    this->beta.insert(this->beta.begin(), c->beta()->begin(), c->beta()->end());
    this->virtual_td_xdesc = c->virtual_td_xdesc();
    this->x_data = reinterpret_cast<const void *>(c->x_data());
    this->virtual_td_ydesc = c->virtual_td_ydesc();
    this->y_data = reinterpret_cast<void *>(c->y_data());
    this->virtual_td_zdesc = c->virtual_td_zdesc();
    this->z_data = reinterpret_cast<const void *>(c->z_data());
    this->virtual_td_bn_scale_bias_mean_var_desc =
        c->virtual_td_bn_scale_bias_mean_var_desc();
    this->bn_scale_data = reinterpret_cast<const void *>(c->bn_scale_data());
    this->bn_bias_data = reinterpret_cast<const void *>(c->bn_bias_data());
    this->exponential_average_factor = c->exponential_average_factor();
    this->result_running_mean_data =
        reinterpret_cast<void *>(c->result_running_mean_data());
    this->result_running_variance_data =
        reinterpret_cast<void *>(c->result_running_variance_data());
    this->epsilon = c->epsilon();
    this->save_mean = reinterpret_cast<void *>(c->save_mean());
    this->save_inv_variance = reinterpret_cast<void *>(c->save_inv_variance());
    this->virtual_ad_activation_desc = c->virtual_ad_activation_desc();
    this->workspace = reinterpret_cast<void *>(c->workspace());
    this->workspace_size_in_bytes = c->workspace_size_in_bytes();
    this->reserve_space = reinterpret_cast<void *>(c->reserve_space());
    this->reserve_space_size_in_bytes = c->workspace_size_in_bytes();
}

uint64_t CudnnBatchNormalizationForwardTrainingEx::executeNative(
    CudaVirtualDevice &vdev) {
    static auto real =
        GET_REAL_FUNCTION(cudnnBatchNormalizationForwardTrainingEx);
    return real(
        vdev.cudnn_handles_virtual_to_real[this->virtual_handle], this->mode,
        this->bn_ops, this->alpha.data(), this->beta.data(),
        vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td_xdesc],
        this->x_data,
        vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td_zdesc],
        this->z_data,
        vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td_ydesc],
        this->y_data,
        vdev.cudnn_tensor_descriptor_virtual_to_real
            [this->virtual_td_bn_scale_bias_mean_var_desc],
        this->bn_scale_data, this->bn_bias_data,
        this->exponential_average_factor, this->result_running_mean_data,
        this->result_running_variance_data, this->epsilon, this->save_mean,
        this->save_inv_variance,
        reinterpret_cast<cudnnActivationDescriptor_t>(
            this->virtual_ad_activation_desc),
        this->workspace, this->workspace_size_in_bytes, this->reserve_space,
        this->reserve_space_size_in_bytes);
}

flatbuffers::Offset<FBCudaApiCall>
CudnnBatchNormalizationForwardTrainingEx::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCudnnBatchNormalizationForwardTrainingEx(
        builder, this->virtual_handle, this->mode, this->bn_ops,
        builder.CreateVector(this->alpha), builder.CreateVector(this->beta),
        this->virtual_td_xdesc, reinterpret_cast<uint64_t>(this->x_data),
        this->virtual_td_ydesc, reinterpret_cast<uint64_t>(this->y_data),
        this->virtual_td_zdesc, reinterpret_cast<uint64_t>(this->z_data),
        this->virtual_td_bn_scale_bias_mean_var_desc,
        reinterpret_cast<uint64_t>(this->bn_scale_data),
        reinterpret_cast<uint64_t>(this->bn_bias_data),
        this->exponential_average_factor,
        reinterpret_cast<uint64_t>(this->result_running_mean_data),
        reinterpret_cast<uint64_t>(this->result_running_variance_data),
        this->epsilon, reinterpret_cast<uint64_t>(this->save_mean),
        reinterpret_cast<uint64_t>(this->save_inv_variance),
        this->virtual_ad_activation_desc,
        reinterpret_cast<uint64_t>(this->workspace),
        this->workspace_size_in_bytes,
        reinterpret_cast<uint64_t>(this->reserve_space),
        this->reserve_space_size_in_bytes);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudnnBatchNormalizationForwardTrainingEx,
        api_call.Union());
    return api_call_union;
}

} // namespace gpuless