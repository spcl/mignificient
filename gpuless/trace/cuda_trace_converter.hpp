#ifndef GPULESS_CUDA_TRACE_CONVERTER_HPP
#define GPULESS_CUDA_TRACE_CONVERTER_HPP

#include "cuda_trace.hpp"
#include "flatbuffers/flatbuffers.h"
#include <vector>

namespace gpuless {

class CudaTraceConverter {
  public:
    static void traceToExecRequest(CudaTrace &cuda_trace,
                                   flatbuffers::FlatBufferBuilder &builder);

    static std::shared_ptr<AbstractCudaApiCall>
    fbAbstractCudaApiCallDeserialize(const FBCudaApiCall *fb_cuda_api_call);

    static std::vector<std::shared_ptr<AbstractCudaApiCall>>
    execRequestToTrace(const FBTraceExecRequest *fb_trace_exec_request);

    static std::shared_ptr<AbstractCudaApiCall>
    execResponseToTopApiCall(const FBTraceExecResponse *fb_trace_exec_response);


};

} // namespace gpuless

#endif // GPULESS_CUDA_TRACE_CONVERTER_HPP