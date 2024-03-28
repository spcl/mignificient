#include "cuda_trace_converter.hpp"
#include "cublas_api_calls.hpp"
#include "cuda_trace.hpp"
#include "cudnn_api_calls.hpp"
#include "manager/manager_device.hpp"
#include <spdlog/spdlog.h>

namespace gpuless {

void CudaTraceConverter::traceToExecRequest(
    CudaTrace &cuda_trace, flatbuffers::FlatBufferBuilder &builder) {
    std::vector<flatbuffers::Offset<FBCudaApiCall>> fb_call_trace;

    for (auto &c : cuda_trace.callStack()) {
        SPDLOG_DEBUG("Serializing api call: {}", c->typeName());
        fb_call_trace.push_back(c->fbSerialize(builder));
    }

    std::vector<flatbuffers::Offset<FBNewModule>> fb_new_modules;
    std::vector<flatbuffers::Offset<FBNewFunction>> fb_new_functions;

    std::set<uint64_t> required_modules;
    std::set<std::string> required_functions;
    for (auto &apiCall : cuda_trace.callStack()) {
        auto rmod_vec = apiCall->requiredCudaModuleIds();
        required_modules.insert(rmod_vec.begin(), rmod_vec.end());
        auto rfunc_vec = apiCall->requiredFunctionSymbols();
        required_functions.insert(rfunc_vec.begin(), rfunc_vec.end());
    }

    for (const auto &rmod_id : required_modules) {
        auto it = cuda_trace.getModuleIdToFatbinResource().find(rmod_id);
        if (it == cuda_trace.getModuleIdToFatbinResource().end()) {
            SPDLOG_ERROR("Required module {} unknown");
        }

        const void *resource_ptr = std::get<0>(it->second);
        uint64_t size = std::get<1>(it->second);
        bool is_loaded = std::get<2>(it->second);

        if (!is_loaded) {
            std::vector<uint8_t> buffer(size);
            std::memcpy(buffer.data(), resource_ptr, size);
            fb_new_modules.push_back(CreateFBNewModule(
                builder, builder.CreateVector(buffer), rmod_id));
            std::get<2>(it->second) = true;
        }
    }

    for (const auto &rfunc : required_functions) {
        auto it = cuda_trace.getSymbolToModuleId().find(rfunc);
        if (it == cuda_trace.getSymbolToModuleId().end()) {
            SPDLOG_ERROR("Required function {} unknown");
        }

        uint64_t module_id = std::get<0>(it->second);
        bool fn_is_loaded = std::get<1>(it->second);

        if (!fn_is_loaded) {
            auto mod_it =
                cuda_trace.getModuleIdToFatbinResource().find(module_id);
            if (mod_it == cuda_trace.getModuleIdToFatbinResource().end()) {
                SPDLOG_ERROR("Unknown module {} for function", module_id,
                             rfunc);
            }

            bool module_is_loaded = std::get<2>(mod_it->second);
            if (!module_is_loaded) {
                SPDLOG_ERROR("Module {} not previously loaded", module_id);
            }

            fb_new_functions.push_back(CreateFBNewFunction(
                builder, builder.CreateString(rfunc), module_id));
            std::get<1>(it->second) = true;
        }
    }

    auto fb_exec_request =
        CreateFBTraceExecRequest(builder, builder.CreateVector(fb_call_trace),
                                 builder.CreateVector(fb_new_modules),
                                 builder.CreateVector(fb_new_functions));

    auto fb_protocol_message = CreateFBProtocolMessage(
        builder, FBMessage_FBTraceExecRequest, fb_exec_request.Union());
    builder.Finish(fb_protocol_message);
}

std::shared_ptr<AbstractCudaApiCall>
CudaTraceConverter::fbAbstractCudaApiCallDeserialize(
    const FBCudaApiCall *fb_cuda_api_call) {
    std::shared_ptr<AbstractCudaApiCall> cuda_api_call;

    switch (fb_cuda_api_call->api_call_type()) {
    case FBCudaApiCallUnion_NONE:
        SPDLOG_ERROR("Cannot convert FBCudaApiCallUnion_NONE");
        std::exit(EXIT_FAILURE);
    // CUDA
    case FBCudaApiCallUnion_FBCudaMalloc:
        cuda_api_call = std::make_shared<CudaMalloc>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudaMemcpyH2D: {
        auto s = std::chrono::high_resolution_clock::now();
        cuda_api_call = std::make_shared<CudaMemcpyH2D>(fb_cuda_api_call);
        auto e = std::chrono::high_resolution_clock::now();
        auto d =
            std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
            1000000.0;
        std::cerr << "h2d deser " << d << std::endl;
        break;
    } case FBCudaApiCallUnion_FBCudaMemcpyD2H: {
        auto s = std::chrono::high_resolution_clock::now();
        cuda_api_call = std::make_shared<CudaMemcpyD2H>(fb_cuda_api_call);
        auto e = std::chrono::high_resolution_clock::now();
        auto d =
            std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
            1000000.0;
        std::cerr << "d2h deser " << d << std::endl;
        break;
    } case FBCudaApiCallUnion_FBCudaMemcpyD2D:
        cuda_api_call = std::make_shared<CudaMemcpyD2D>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudaMemcpyAsyncH2D:
        cuda_api_call = std::make_shared<CudaMemcpyAsyncH2D>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudaMemcpyAsyncD2H:
        cuda_api_call = std::make_shared<CudaMemcpyAsyncD2H>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudaMemcpyAsyncD2D:
        cuda_api_call = std::make_shared<CudaMemcpyAsyncD2D>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudaLaunchKernel:
        cuda_api_call = std::make_shared<CudaLaunchKernel>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudaFree:
        cuda_api_call = std::make_shared<CudaFree>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudaStreamSynchronize:
        cuda_api_call =
            std::make_shared<CudaStreamSynchronize>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudaGetDeviceProperties:
        cuda_api_call =
            std::make_shared<CudaGetDeviceProperties>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudaDeviceSynchronize:
        cuda_api_call =
            std::make_shared<CudaDeviceSynchronize>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudaFuncGetAttributes:
        cuda_api_call =
            std::make_shared<CudaFuncGetAttributes>(fb_cuda_api_call);
        break;

    // cuBLAS
    case FBCudaApiCallUnion_FBCublasCreateV2:
        cuda_api_call = std::make_shared<CublasCreateV2>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCublasSetStreamV2:
        cuda_api_call = std::make_shared<CublasSetStreamV2>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCublasSetMathMode:
        cuda_api_call = std::make_shared<CublasSetMathMode>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCublasSgemmV2:
        cuda_api_call = std::make_shared<CublasSgemmV2>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCublasLtCreate:
        cuda_api_call = std::make_shared<CublasLtCreate>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCublasLtMatmulDescCreate:
        cuda_api_call =
            std::make_shared<CublasLtMatmulDescCreate>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCublasLtMatmulDescDestroy:
        cuda_api_call =
            std::make_shared<CublasLtMatmulDescDestroy>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCublasLtMatmulDescSetAttribute:
        cuda_api_call =
            std::make_shared<CublasLtMatmulDescSetAttribute>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCublasLtMatmul:
        cuda_api_call = std::make_shared<CublasLtMatmul>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCublasLtMatrixLayoutCreate:
        cuda_api_call =
            std::make_shared<CublasLtMatrixLayoutCreate>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCublasLtMatrixLayoutDestroy:
        cuda_api_call =
            std::make_shared<CublasLtMatrixLayoutDestroy>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCublasLtMatrixLayoutSetAttribute:
        cuda_api_call = std::make_shared<CublasLtMatrixLayoutSetAttribute>(
            fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCublasSgemmStridedBatched:
        cuda_api_call =
            std::make_shared<CublasSgemmStridedBatched>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCublasLtMatmulAlgoGetHeuristic:
        cuda_api_call =
            std::make_shared<CublasLtMatmulAlgoGetHeuristic>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCublasLtMatmulPreferenceDestroy:
        cuda_api_call =
            std::make_shared<CublasLtMatmulPreferenceDestroy>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCublasLtMatmulPreferenceSetAttribute:
        cuda_api_call =
            std::make_shared<CublasLtMatmulPreferenceSetAttribute>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCublasLtMatmulPreferenceCreate:
        cuda_api_call =
            std::make_shared<CublasLtMatmulPreferenceCreate>(fb_cuda_api_call);
        break;

    // cuDNN
    case FBCudaApiCallUnion_FBCudnnCreate:
        cuda_api_call = std::make_shared<CudnnCreate>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudnnSetStream:
        cuda_api_call = std::make_shared<CudnnSetStream>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudnnCreateTensorDescriptor:
        cuda_api_call =
            std::make_shared<CudnnCreateTensorDescriptor>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudnnSetTensorNdDescriptor:
        cuda_api_call =
            std::make_shared<CudnnSetTensorNdDescriptor>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudnnCreateFilterDescriptor:
        cuda_api_call =
            std::make_shared<CudnnCreateFilterDescriptor>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudnnSetFilterNdDescriptor:
        cuda_api_call =
            std::make_shared<CudnnSetFilterNdDescriptor>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudnnCreateConvolutionDescriptor:
        cuda_api_call = std::make_shared<CudnnCreateConvolutionDescriptor>(
            fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudnnSetConvolutionGroupCount:
        cuda_api_call =
            std::make_shared<CudnnSetConvolutionGroupCount>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudnnSetConvolutionMathType:
        cuda_api_call =
            std::make_shared<CudnnSetConvolutionMathType>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudnnSetConvolutionNdDescriptor:
        cuda_api_call =
            std::make_shared<CudnnSetConvolutionNdDescriptor>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudnnGetConvolutionForwardAlgorithmV7:
        cuda_api_call = std::make_shared<CudnnGetConvolutionForwardAlgorithmV7>(
            fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudnnConvolutionForward:
        cuda_api_call =
            std::make_shared<CudnnConvolutionForward>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudnnBatchNormalizationForwardInference:
        cuda_api_call =
            std::make_shared<CudnnBatchNormalizationForwardInference>(
                fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudnnDestroyConvolutionDescriptor:
        cuda_api_call = std::make_shared<CudnnDestroyConvolutionDescriptor>(
            fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudnnDestroyFilterDescriptor:
        cuda_api_call =
            std::make_shared<CudnnDestroyFilterDescriptor>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudnnDestroyTensorDescriptor:
        cuda_api_call =
            std::make_shared<CudnnDestroyTensorDescriptor>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudnnConvolutionBackwardData:
        cuda_api_call =
            std::make_shared<CudnnConvolutionBackwardData>(fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudnnGetConvolutionBackwardDataAlgorithmV7:
        cuda_api_call =
            std::make_shared<CudnnGetConvolutionBackwardDataAlgorithmV7>(
                fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudnnGetBatchNormalizationForwardTrainingExWorkspaceSize:
        cuda_api_call = std::make_shared<
            CudnnGetBatchNormalizationForwardTrainingExWorkspaceSize>(
            fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudnnGetBatchNormalizationTrainingExReserveSpaceSize:
        cuda_api_call = std::make_shared<
            CudnnGetBatchNormalizationTrainingExReserveSpaceSize>(
            fb_cuda_api_call);
        break;
    case FBCudaApiCallUnion_FBCudnnBatchNormalizationForwardTrainingEx:
        cuda_api_call =
            std::make_shared<CudnnBatchNormalizationForwardTrainingEx>(
                fb_cuda_api_call);
        break;
    }

    return cuda_api_call;
}

std::shared_ptr<AbstractCudaApiCall>
CudaTraceConverter::execResponseToTopApiCall(
    const FBTraceExecResponse *fb_trace_exec_response) {
    return CudaTraceConverter::fbAbstractCudaApiCallDeserialize(
        fb_trace_exec_response->trace_top());
}

std::vector<std::shared_ptr<AbstractCudaApiCall>>
CudaTraceConverter::execRequestToTrace(
    const FBTraceExecRequest *fb_trace_exec_request) {
    std::vector<std::shared_ptr<AbstractCudaApiCall>> cuda_api_calls;

    for (const auto &c : *fb_trace_exec_request->trace()) {
        cuda_api_calls.push_back(
            CudaTraceConverter::fbAbstractCudaApiCallDeserialize(c));
    }

    return cuda_api_calls;
}

} // namespace gpuless
