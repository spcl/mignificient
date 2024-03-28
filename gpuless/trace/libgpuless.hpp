#ifndef __LIBGPULESS_HPP__
#define __LIBGPULESS_HPP__

#include <iostream>
#include <stack>

#include <cuda.h>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>
#include <unistd.h>

#include "cuda_trace.hpp"
#include "dlsym_util.hpp"
#include "trace_executor.hpp"

#define LINK_CU_FUNCTION(symbol, f)                                            \
    do {                                                                       \
        if (strcmp(symbol, #f) == 0) {                                         \
            *pfn = (void *)&f;                                                 \
            return CUDA_SUCCESS;                                               \
        }                                                                      \
    } while (0)

#define LINK_CU_FUNCTION_DLSYM(symbol, f)                                      \
    do {                                                                       \
        if (strcmp(symbol, #f) == 0) {                                         \
            return (void *)&f;                                                 \
        }                                                                      \
    } while (0)

#define GET_REAL_FUNCTION(fn) (decltype(&fn))real_dlsym(RTLD_NEXT, #fn)

#define HIJACK_FN_PROLOGUE()                                                   \
    do {                                                                       \
        SPDLOG_INFO("{}() [pid={}]", __func__, getpid());                      \
    } while (0)

#define EXIT_NOT_IMPLEMENTED(fn)                                               \
    do {                                                                       \
        SPDLOG_ERROR("not implemented: {}", fn);                               \
        std::exit(EXIT_FAILURE);                                               \
    } while (0)

#define EXIT_UNRECOVERABLE(msg)                                                \
    do {                                                                       \
        std::cerr << msg << std::endl;                                         \
        std::cerr << "unrecoverable error, exiting" << std::endl;              \
        std::exit(EXIT_FAILURE);                                               \
    } while (0)

struct CudaCallConfig {
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem{};
    struct CUstream_st *stream{};
};

struct CudaRegisterState {
    uint64_t current_fatbin_handle;
    bool is_registering;
};

gpuless::CudaTrace &getCudaTrace();
std::shared_ptr<gpuless::TraceExecutor> getTraceExecutor(bool clean = false);

#endif // __LIBGPULESS_HPP__
