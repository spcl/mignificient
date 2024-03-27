#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cublas.h>
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <dlfcn.h>
#include <iomanip>
#include <spdlog/cfg/env.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <sstream>
#include <string>
#include <vector>

#include "../trace/cubin_analysis.hpp"
#include "../utils.hpp"

auto get_logger() {
    static auto logger =
        spdlog::basic_logger_mt("arg_trace", "logs/arg_trace.log");
    static bool init = false;
    if (!init) {
        spdlog::set_pattern("[%H:%M:%S:%e:%f] %v");
        init = true;
    }
    return logger;
}

#define LINK_CU_FUNCTION(symbol, f)                                            \
    do {                                                                       \
        if (strcmp(symbol, #f) == 0) {                                         \
            *pfn = (void *)&f;                                                 \
            return CUDA_SUCCESS;                                               \
        }                                                                      \
    } while (0)

#define GET_REAL_FUNCTION(fn) (decltype(&fn))real_dlsym(RTLD_NEXT, #fn)

const bool MAP_PTR_TO_IDENT = true;

const int major_compute_version = 8;
const int minor_compute_version = 6;

static double acc_time = 0.0;

// execute a command in a shell
static std::string exec(const char *cmd) {
    std::array<char, 128> buffer{};
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

void *real_dlsym(void *handle, const char *symbol) {
    static auto internal_dlsym =
        (decltype(&dlsym))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
    return (*internal_dlsym)(handle, symbol);
}

std::map<void *, std::string> &getCUfnptrToSymbolMap() {
    static std::map<void *, std::string> cufnptr_to_symbol_map;
    return cufnptr_to_symbol_map;
}

// Dummy parameter-pack expander
template <class T> void expand(std::initializer_list<T>) {}

// PRINTING HELPER
std::ostream &operator<<(std::ostream &os, const dim3 &dim) {
    os << "(" << dim.x << " " << dim.y << " " << dim.z << ")";
    return os;
}

template <class Fun, class... Args>
decltype(auto) trace_args(const char *symbol, Args &&...args) {

    // Print Function with parameters:
    std::stringstream ss;
    ss << symbol;
    if constexpr (sizeof...(args)) {
        ss << " ( ";
        expand({(ss << std::get<0>(args) << ":{" << std::get<1>(args) << "} ",
                 0)...});
        ss << ")\n";
    } else {
        ss << "( )";
    }

    SPDLOG_LOGGER_INFO(get_logger(), ss.str());

    // Forward the call
    auto real_func = (Fun)real_dlsym(RTLD_NEXT, symbol);
    return real_func(std::get<1>(args)...);
}

std::string addrToIdentDevice(void *addr) {
    static std::map<void *, std::string> addr_to_ident;
    static int ctr = 0;
    const char *prefix = "D";

    std::string ident;
    auto it = addr_to_ident.find(addr);
    if (it != addr_to_ident.end()) {
        ident = it->second;
    } else {
        std::stringstream ss;
        ss << prefix << ctr;
        addr_to_ident[addr] = ss.str();
        ctr++;
        ident = ss.str();
    }

    return ident;
}

std::string addrToIdentHost(void *addr) {
    static std::map<void *, std::string> addr_to_ident;
    static int ctr = 0;
    const char *prefix = "H";

    std::string ident;
    auto it = addr_to_ident.find(addr);
    if (it != addr_to_ident.end()) {
        ident = it->second;
    } else {
        std::stringstream ss;
        ss << prefix << ctr;
        addr_to_ident[addr] = ss.str();
        ctr++;
        ident = ss.str();
    }

    return ident;
}

CubinAnalyzer &getAnalyzer() {
    static CubinAnalyzer analyzer;
    static bool initialized = false;

    if (!initialized) {
        char *cuda_binary = std::getenv("CUDA_BINARY");
        if (cuda_binary == nullptr) {
            std::cerr << "[error] please set CUDA_BINARY environment variable"
                      << std::endl;
            return analyzer;
        }

        std::vector<std::string> cuda_binaries;
        string_split(std::string(cuda_binary), ',', cuda_binaries);
        analyzer.analyze(cuda_binaries, major_compute_version,
                         minor_compute_version);
        initialized = true;
    }

    return analyzer;
}

static int syncs = 0;

#define SYNCHRONIZE()                                                          \
    do {                                                                       \
        std::stringstream syncstream;                                          \
        syncstream << "Synchronization " << syncs++ << ".";                    \
        SPDLOG_LOGGER_INFO(get_logger(), syncstream.str());                    \
    } while (0)

// Make a FOREACH macro
#define FE_0(WHAT)
#define FE_1(WHAT, X) WHAT(X)
#define FE_2(WHAT, X, ...) WHAT(X), FE_1(WHAT, __VA_ARGS__)
#define FE_3(WHAT, X, ...) WHAT(X), FE_2(WHAT, __VA_ARGS__)
#define FE_4(WHAT, X, ...) WHAT(X), FE_3(WHAT, __VA_ARGS__)
#define FE_5(WHAT, X, ...) WHAT(X), FE_4(WHAT, __VA_ARGS__)
#define FE_6(WHAT, X, ...) WHAT(X), FE_5(WHAT, __VA_ARGS__)
#define FE_7(WHAT, X, ...) WHAT(X), FE_6(WHAT, __VA_ARGS__)
#define FE_8(WHAT, X, ...) WHAT(X), FE_7(WHAT, __VA_ARGS__)
#define FE_9(WHAT, X, ...) WHAT(X), FE_8(WHAT, __VA_ARGS__)
#define FE_10(WHAT, X, ...) WHAT(X), FE_9(WHAT, __VA_ARGS__)
#define FE_11(WHAT, X, ...) WHAT(X), FE_10(WHAT, __VA_ARGS__)
#define FE_12(WHAT, X, ...) WHAT(X), FE_11(WHAT, __VA_ARGS__)
#define FE_13(WHAT, X, ...) WHAT(X), FE_12(WHAT, __VA_ARGS__)
#define FE_14(WHAT, X, ...) WHAT(X), FE_13(WHAT, __VA_ARGS__)
#define FE_15(WHAT, X, ...) WHAT(X), FE_14(WHAT, __VA_ARGS__)
#define FE_16(WHAT, X, ...) WHAT(X), FE_15(WHAT, __VA_ARGS__)
#define FE_17(WHAT, X, ...) WHAT(X), FE_16(WHAT, __VA_ARGS__)
#define FE_18(WHAT, X, ...) WHAT(X), FE_17(WHAT, __VA_ARGS__)
#define FE_19(WHAT, X, ...) WHAT(X), FE_18(WHAT, __VA_ARGS__)
#define FE_20(WHAT, X, ...) WHAT(X), FE_19(WHAT, __VA_ARGS__)
#define FE_21(WHAT, X, ...) WHAT(X), FE_20(WHAT, __VA_ARGS__)
#define FE_22(WHAT, X, ...) WHAT(X), FE_21(WHAT, __VA_ARGS__)
#define FE_23(WHAT, X, ...) WHAT(X), FE_22(WHAT, __VA_ARGS__)
#define FE_24(WHAT, X, ...) WHAT(X), FE_23(WHAT, __VA_ARGS__)
#define FE_25(WHAT, X, ...) WHAT(X), FE_24(WHAT, __VA_ARGS__)

#define GET_MACRO(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13,  \
                  _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25,  \
                  NAME, ...)                                                   \
    NAME
#define FOR_EACH(action, ...)                                                  \
    GET_MACRO(_0, __VA_ARGS__, FE_25, FE_24, FE_23, FE_22, FE_21, FE_20,       \
              FE_19, FE_18, FE_17, FE_16, FE_15, FE_14, FE_13, FE_12, FE_11,   \
              FE_10, FE_9, FE_8, FE_7, FE_6, FE_5, FE_4, FE_3, FE_2, FE_1,     \
              FE_0)                                                            \
    (action, __VA_ARGS__)

#define TYPE_MACRO(x) decltype(x)
#define NAME_MACRO(x) std::make_tuple(#x, x)

#define TRACE_FUNC(f, ...)                                                     \
    trace_args<decltype(f(__VA_ARGS__)) (*)(FOR_EACH(                          \
        TYPE_MACRO, ##__VA_ARGS__))>(#f, FOR_EACH(NAME_MACRO, ##__VA_ARGS__))

#define TRACE_FUNC_NARGS(f) trace_args<decltype(f()) (*)()>(#f)
// FUNCTION TRACES
extern "C" {

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
                             const char *name) {
    static auto real_func =
        (decltype(&cuModuleGetFunction))real_dlsym(RTLD_NEXT, __func__);
    auto err = real_func(hfunc, hmod, name);
    auto &map = getCUfnptrToSymbolMap();
    map.emplace(std::make_pair((void *)*hfunc, name));
    return err;
}

extern "C" CUresult CUDAAPI cuLaunchKernel(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams, void **extra) {
    static auto real_func =
        (decltype(&cuLaunchKernel))real_dlsym(RTLD_NEXT, __func__);
    auto analyzer = getAnalyzer();

    std::stringstream ss;
    ss << "cuLaunchKernel(";

    auto &map = getCUfnptrToSymbolMap();
    auto it = map.find((void *)f);
    if (it != map.end()) {
        // Symbol Name
        std::string &kernel_symbol = it->second;
        std::string cmd = "echo " + kernel_symbol + "| c++filt";
        auto demangled_symbol = exec(cmd.c_str());
        ss << string_rstrip(demangled_symbol);
        std::vector<KParamInfo> params;

        // grid/block configuaration
        ss << "grid(";
        ss << gridDimX << ",";
        ss << gridDimY << ",";
        ss << gridDimZ;
        ss << "),block(";
        ss << blockDimX << ",";
        ss << blockDimY << ",";
        ss << blockDimZ;
        ss << ")" << std::endl;

        // stream
        ss << "sharedMem: " << sharedMemBytes << std::endl;
        ss << "stream: " << hStream << std::endl;

        // parameters
        ss << "args ( ";
        for (int i = 0; i < params.size(); i++) {
            const auto &p = params[i];
            void *d_ptr = *((void **)kernelParams[i]);
            ss << d_ptr;

            if (i < params.size() - 1) {
                ss << ", ";
            }
        }
        ss << " )" << std::endl;

    } else {
        ss << "unknown kernel";
    }

    ss << ")";

    auto err =
        real_func(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                  blockDimZ, sharedMemBytes, hStream, kernelParams, extra);

    SPDLOG_LOGGER_INFO(get_logger(), ss.str());

    return err;
}

extern "C" void CUDARTAPI __cudaRegisterFunction(
    void **fatCubinHandle, const char *hostFun, char *deviceFun,
    const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid,
    dim3 *bDim, dim3 *gDim, int *wSize) {
    static auto real_func =
        (decltype(&__cudaRegisterFunction))real_dlsym(RTLD_NEXT, __func__);
    auto &map = getCUfnptrToSymbolMap();
    map.emplace(std::make_pair((void *)deviceFun, deviceName));
    map.emplace(std::make_pair((void *)hostFun, deviceName));
    real_func(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid,
              bid, bDim, gDim, wSize);
}

cudaError_t CUDARTAPI cudaLaunchKernel(const void *func, dim3 gridDim,
                                       dim3 blockDim, void **args,
                                       size_t sharedMem, cudaStream_t stream) {
    static auto real_func =
        (cudaError_t(*)(const void *, dim3, dim3, void **, size_t,
                        cudaStream_t))real_dlsym(RTLD_NEXT, __func__);

    cudaError_t err;
    auto analyzer = getAnalyzer();

    // get kernel parameters
    auto &map = getCUfnptrToSymbolMap();
    auto it = map.find((void *)func);
    if (it != map.end()) {
        err = real_func(func, gridDim, blockDim, args, sharedMem, stream);

        std::string &kernel_symbol = it->second;
        std::vector<KParamInfo> params;
        analyzer.kernel_parameters(kernel_symbol, params);

        std::stringstream ss;

        ss << "cudaLaunchKernel( ";

        // kernel symbol name
        std::string cmd = "echo " + kernel_symbol + "| c++filt";
        auto demangled_symbol = exec(cmd.c_str());
        ss << "name{" << string_rstrip(demangled_symbol) << "}";

        // grid/block configuaration
        ss << "grid{";
        ss << gridDim.x << ",";
        ss << gridDim.y << ",";
        ss << gridDim.z;
        ss << "},block{";
        ss << blockDim.x << ",";
        ss << blockDim.y << ",";
        ss << blockDim.z;
        ss << "} ";

        // stream
        ss << "sharedMem {" << sharedMem << "} ";
        ss << "stream {" << stream << "} ";

        // parameters
        ss << "args {";
        for (int i = 0; i < params.size(); i++) {
            const auto &p = params[i];
            void *d_ptr = *((void **)args[i]);
            ss << std::hex << d_ptr;

            if (i < params.size() - 1) {
                ss << ", ";
            }
        }
        ss << "} ";
        ss << " )" << std::endl;

        SPDLOG_LOGGER_INFO(get_logger(), ss.str());
    } else {
        std::cerr << "[error] no symbol found" << std::endl;
    }

    return err;
}

cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count,
                                 enum cudaMemcpyKind kind) {
    static auto real_func =
        (decltype(&cudaMemcpy))real_dlsym(RTLD_NEXT, __func__);

    std::stringstream ss;
    std::string kind_str;
    if (kind == cudaMemcpyKind::cudaMemcpyDeviceToHost) {
        kind_str = "cudaMemcpyDeviceToHost";
        ss << "cudaMemcpy(" << kind_str << ", ";
        ss << addrToIdentHost(dst) << "(" << std::hex << dst << ")"
           << " <- " << addrToIdentDevice((void *)src) << "(" << std::hex << src
           << ")"
           << ", " << count << " bytes)";
        SYNCHRONIZE();
    } else if (kind == cudaMemcpyKind::cudaMemcpyHostToDevice) {
        kind_str = "cudaMemcpyHostToDevice";
        ss << "cudaMemcpy(" << kind_str << ") ";
        ss << addrToIdentDevice(dst) << "(" << std::hex << dst << ")"
           << " <- " << addrToIdentHost((void *)src) << "(" << std::hex << src
           << ")"
           << ", " << count << " bytes)";
    } else if (kind == cudaMemcpyKind::cudaMemcpyDeviceToDevice) {
        kind_str = "cudaMemcpyDeviceToDevice";
        ss << "cudaMemcpy(" << kind_str << ") ";
        ss << addrToIdentDevice(dst) << " <- " << addrToIdentDevice((void *)src)
           << ")"
           << ", " << count << " bytes)";
    }
    ss << std::endl;
    SPDLOG_LOGGER_INFO(get_logger(), ss.str());
    return real_func(dst, src, count, kind);
}

cudnnStatus_t cudnnCreate(cudnnHandle_t *handle) {
    return TRACE_FUNC(cudnnCreate, handle);
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    using T = cudaError_t (*)(void **, size_t);
    static auto real_func = (T)real_dlsym(RTLD_NEXT, __func__);
    std::stringstream ss;
    ss << "cudaMalloc(" << addrToIdentDevice(*devPtr) << " (" << std::hex
       << *devPtr << "), " << size << " bytes)" << std::endl;
    SYNCHRONIZE();
    SPDLOG_LOGGER_INFO(get_logger(), ss.str());
    return real_func(devPtr, size);
}

cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t hStream) {
    using T = cudaError_t (*)(void **, size_t, cudaStream_t);
    static auto real_func = (T)real_dlsym(RTLD_NEXT, __func__);

    std::stringstream ss;
    ss << "cudaMallocAsync(" << addrToIdentDevice(*devPtr) << " (" << std::hex
       << *devPtr << "), " << size << " bytes"
       << ", stream " << ((uint64_t)hStream) << ")" << std::endl;
    SPDLOG_LOGGER_INFO(get_logger(), ss.str());
    return real_func(devPtr, size, hStream);
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            enum cudaMemcpyKind kind, cudaStream_t stream) {
    static auto real_func =
        (decltype(&cudaMemcpyAsync))real_dlsym(RTLD_NEXT, __func__);

    std::stringstream ss;
    std::string kind_str;
    if (kind == cudaMemcpyKind::cudaMemcpyDeviceToHost) {
        kind_str = "cudaMemcpyDeviceToHost";
        ss << "cudaMemcpyAsync(" << kind_str << ", ";
        ss << addrToIdentHost(dst) << "(" << std::hex << dst << ")"
           << " <- " << addrToIdentDevice((void *)src) << "(" << std::hex << src
           << ")"
           << ", " << count << " bytes)";
        SYNCHRONIZE();
    } else if (kind == cudaMemcpyKind::cudaMemcpyHostToDevice) {
        kind_str = "cudaMemcpyHostToDevice";
        ss << "cudaMemcpyAsync(" << kind_str << ") ";
        ss << addrToIdentDevice(dst) << "(" << std::hex << dst << ")"
           << " <- " << addrToIdentHost((void *)src) << "(" << std::hex << src
           << ")"
           << ", " << count << " bytes)";
    } else if (kind == cudaMemcpyKind::cudaMemcpyDeviceToDevice) {
        kind_str = "cudaMemcpyDeviceToDevice";
        ss << "cudaMemcpyAsync(" << kind_str << ") ";
        ss << addrToIdentDevice(dst) << " <- " << addrToIdentDevice((void *)src)
           << ")"
           << ", " << count << " bytes)";
    }
    ss << std::endl;
    SPDLOG_LOGGER_INFO(get_logger(), ss.str());

    return real_func(dst, src, count, kind, stream);
}

cudaError_t cudaFree(void *devPtr) {
    SYNCHRONIZE();
    return TRACE_FUNC(cudaFree, devPtr);
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    return TRACE_FUNC(cudaStreamSynchronize, stream);
}

cudaError_t cudaThreadSynchronize(void) {
    return TRACE_FUNC_NARGS(cudaThreadSynchronize);
}

cudaError_t cudaDeviceSynchronize(void) {
    return TRACE_FUNC_NARGS(cudaDeviceSynchronize);
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
    SYNCHRONIZE();
    return TRACE_FUNC(cudaGetDeviceProperties, prop, device);
}

cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr,
                                   int device) {
    return TRACE_FUNC(cudaDeviceGetAttribute, value, attr, device);
}

cudaError_t cudaFuncGetAttributes(cudaFuncAttributes *attr, const void *func) {
    SYNCHRONIZE();
    return TRACE_FUNC(cudaFuncGetAttributes, attr, func);
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, const void *func, int blockSize, size_t dynamicSmemSize,
    unsigned int flags) {
    return TRACE_FUNC(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
                      numBlocks, func, blockSize, dynamicSmemSize, flags);
}

CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    SYNCHRONIZE();
    return TRACE_FUNC(cuDevicePrimaryCtxRelease, dev);
}

cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId) {
    return TRACE_FUNC(cudnnSetStream, handle, streamId);
}

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc) {
    return TRACE_FUNC(cudnnCreateTensorDescriptor, tensorDesc);
}

cudnnStatus_t cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                         cudnnDataType_t dataType, int nbDims,
                                         const int dimA[],
                                         const int strideA[]) {
    return TRACE_FUNC(cudnnSetTensorNdDescriptor, tensorDesc, dataType, nbDims,
                      dimA, strideA);
}

cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc) {
    return TRACE_FUNC(cudnnCreateFilterDescriptor, filterDesc);
}

cudnnStatus_t cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t filterDesc,
                                         cudnnDataType_t dataType,
                                         cudnnTensorFormat_t format, int nbDims,
                                         const int filterDimA[]) {
    return TRACE_FUNC(cudnnSetFilterNdDescriptor, filterDesc, dataType, format,
                      nbDims, filterDimA);
}

cudnnStatus_t
cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc) {
    return TRACE_FUNC(cudnnCreateConvolutionDescriptor, convDesc);
}

cudnnStatus_t
cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc,
                              int groupCount) {
    return TRACE_FUNC(cudnnSetConvolutionGroupCount, convDesc, groupCount);
}

cudnnStatus_t cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc,
                                          cudnnMathType_t mathType) {
    return TRACE_FUNC(cudnnSetConvolutionMathType, convDesc, mathType);
}

cudnnStatus_t cudnnSetConvolutionNdDescriptor(
    cudnnConvolutionDescriptor_t convDesc, int arrayLength, const int padA[],
    const int filterStrideA[], const int dilationA[],
    cudnnConvolutionMode_t mode, cudnnDataType_t dataType) {
    return TRACE_FUNC(cudnnSetConvolutionNdDescriptor, convDesc, arrayLength,
                      padA, filterStrideA, dilationA, mode, dataType);
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults) {
    SYNCHRONIZE();
    auto err = TRACE_FUNC(cudnnGetConvolutionForwardAlgorithm_v7, handle, xDesc,
                      wDesc, convDesc, yDesc, requestedAlgoCount,
                      returnedAlgoCount, perfResults);
    std::stringstream argstream;
    argstream << "Number of algs: " << *returnedAlgoCount << ".";
    argstream << " First alg: "  << perfResults[0].algo << ".";
    SPDLOG_LOGGER_INFO(get_logger(), argstream.str());
    return err;
}

cudnnStatus_t
cudnnConvolutionForward(cudnnHandle_t handle, const void *alpha,
                        const cudnnTensorDescriptor_t xDesc, const void *x,
                        const cudnnFilterDescriptor_t wDesc, const void *w,
                        const cudnnConvolutionDescriptor_t convDesc,
                        cudnnConvolutionFwdAlgo_t algo, void *workSpace,
                        size_t workSpaceSizeInBytes, const void *beta,
                        const cudnnTensorDescriptor_t yDesc, void *y) {
    return TRACE_FUNC(cudnnConvolutionForward, handle, alpha, xDesc, x, wDesc,
                      w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta,
                      yDesc, y);
}

cudnnStatus_t cudnnConvolutionBackwardData(
    cudnnHandle_t handle, const void *alpha,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {
    return TRACE_FUNC(cudnnConvolutionBackwardData, handle, alpha, wDesc, w,
                      dyDesc, dy, convDesc, algo, workSpace,
                      workSpaceSizeInBytes, beta, dxDesc, dx);
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults) {
    SYNCHRONIZE();
    auto err = TRACE_FUNC(cudnnGetConvolutionBackwardDataAlgorithm_v7, handle,
                      wDesc, dyDesc, convDesc, dxDesc, requestedAlgoCount,
                      returnedAlgoCount, perfResults);
    std::stringstream argstream;
    argstream << "Number of algs: " << *returnedAlgoCount << ".";
    argstream << " First alg: "  << perfResults[0].algo << ".";
    SPDLOG_LOGGER_INFO(get_logger(), argstream.str());
    return err;
}

cudnnStatus_t cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t zDesc,
    const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const cudnnActivationDescriptor_t activationDesc, size_t *sizeInBytes) {
    SYNCHRONIZE();
    return TRACE_FUNC(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize,
                      handle, mode, bnOps, xDesc, zDesc, yDesc,
                      bnScaleBiasMeanVarDesc, activationDesc, sizeInBytes);
}

cudnnStatus_t cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes) {
    return TRACE_FUNC(cudnnGetBatchNormalizationTrainingExReserveSpaceSize,
                      handle, mode, bnOps, activationDesc, xDesc, sizeInBytes);
}

cudnnStatus_t cudnnBatchNormalizationForwardInference(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha,
    const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, const void *estimatedMean,
    const void *estimatedVariance, double epsilon) {
    return TRACE_FUNC(cudnnBatchNormalizationForwardInference, handle, mode,
                      alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc,
                      bnScale, bnBias, estimatedMean, estimatedVariance,
                      epsilon);
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
    return TRACE_FUNC(
        cudnnBatchNormalizationForwardTrainingEx, handle, mode, bnOps, alpha,
        beta, xDesc, xData, zDesc, zData, yDesc, yData, bnScaleBiasMeanVarDesc,
        bnScaleData, bnBiasData, exponentialAverageFactor,
        resultRunningMeanData, resultRunningVarianceData, epsilon, saveMean,
        saveInvVariance, activationDesc, workspace, workSpaceSizeInBytes,
        reserveSpace, reserveSpaceSizeInBytes);
}

cudnnStatus_t
cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc) {
    return TRACE_FUNC(cudnnDestroyConvolutionDescriptor, convDesc);
}

cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc) {
    return TRACE_FUNC(cudnnDestroyFilterDescriptor, filterDesc);
}

cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {
    return TRACE_FUNC(cudnnDestroyTensorDescriptor, tensorDesc);
}

cublasStatus_t cublasCreate_v2(cublasHandle_t *handle) {
    return TRACE_FUNC(cublasCreate_v2, handle);
}

cublasStatus_t cublasLtCreate(cublasLtHandle_t *lighthandle) {
    return TRACE_FUNC(cublasLtCreate, lighthandle);
}

cublasStatus_t cublasSetStream_v2(cublasHandle_t handle,
                                  cudaStream_t streamId) {
    return TRACE_FUNC(cublasSetStream_v2, handle, streamId);
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
    return TRACE_FUNC(cublasSetMathMode, handle, mode);
}

cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const float *alpha, const float *A, int lda,
                              const float *B, int ldb, const float *beta,
                              float *C, int ldc) {
    return TRACE_FUNC(cublasSgemm_v2, handle, transa, transb, m, n, k, alpha, A,
                      lda, B, ldb, beta, C, ldc);
}

cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc,
                                        cublasComputeType_t computeType,
                                        cudaDataType_t scaleType) {
    return TRACE_FUNC(cublasLtMatmulDescCreate, matmulDesc, computeType,
                      scaleType);
}

cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc) {
    return TRACE_FUNC(cublasLtMatmulDescDestroy, matmulDesc);
}

cublasStatus_t
cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc,
                               cublasLtMatmulDescAttributes_t attr,
                               const void *buf, size_t sizeInBytes) {
    return TRACE_FUNC(cublasLtMatmulDescSetAttribute, matmulDesc, attr, buf,
                      sizeInBytes);
}

cublasStatus_t
cublasLtMatmul(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc,
               const void *alpha, const void *A, cublasLtMatrixLayout_t Adesc,
               const void *B, cublasLtMatrixLayout_t Bdesc, const void *beta,
               const void *C, cublasLtMatrixLayout_t Cdesc, void *D,
               cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t *algo,
               void *workspace, size_t workspaceSizeInBytes,
               cudaStream_t stream) {
    return TRACE_FUNC(cublasLtMatmul, lightHandle, computeDesc, alpha, A, Adesc,
                      B, Bdesc, beta, C, Cdesc, D, Ddesc, algo, workspace,
                      workspaceSizeInBytes, stream);
}

cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout,
                                          cudaDataType type, uint64_t rows,
                                          uint64_t cols, int64_t ld) {
    return TRACE_FUNC(cublasLtMatrixLayoutCreate, matLayout, type, rows, cols,
                      ld);
}

cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout) {
    return TRACE_FUNC(cublasLtMatrixLayoutDestroy, matLayout);
}

cublasStatus_t
cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout,
                                 cublasLtMatrixLayoutAttribute_t attr,
                                 const void *buf, size_t sizeInBytes) {
    return TRACE_FUNC(cublasLtMatrixLayoutSetAttribute, matLayout, attr, buf,
                      sizeInBytes);
}

cublasStatus_t
cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                          cublasOperation_t transb, int m, int n, int k,
                          const float *alpha, const float *A, int lda,
                          long long int strideA, const float *B, int ldb,
                          long long int strideB, const float *beta, float *C,
                          int ldc, long long int strideC, int batchCount) {
    return TRACE_FUNC(cublasSgemmStridedBatched, handle, transa, transb, m, n,
                      k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc,
                      strideC, batchCount);
}

cublasStatus_t cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc,
    cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc,
    cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc,
    cublasLtMatmulPreference_t preference, int requestedAlgoCount,
    cublasLtMatmulHeuristicResult_t heuristicResultsArray[],
    int *returnAlgoCount) {
    return TRACE_FUNC(cublasLtMatmulAlgoGetHeuristic, lightHandle,
                      operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference,
                      requestedAlgoCount, heuristicResultsArray,
                      returnAlgoCount);
}

cublasStatus_t
cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref) {
    return TRACE_FUNC(cublasLtMatmulPreferenceCreate, pref);
}

cublasStatus_t
cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref,
                                     cublasLtMatmulPreferenceAttributes_t attr,
                                     const void *buf, size_t sizeInBytes) {
    return TRACE_FUNC(cublasLtMatmulPreferenceSetAttribute, pref, attr, buf,
                      sizeInBytes);
}

cublasStatus_t
cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref) {
    return TRACE_FUNC(cublasLtMatmulPreferenceDestroy, pref);
}

extern "C" CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn,
                                             int cudaVersion,
                                             cuuint64_t flags) {
    static auto real_func =
        (decltype(&cuGetProcAddress))real_dlsym(RTLD_NEXT, __func__);

    LINK_CU_FUNCTION(symbol, cuLaunchKernel);
    LINK_CU_FUNCTION(symbol, cuModuleGetFunction);

    return real_func(symbol, pfn, cudaVersion, flags);
}
}

extern "C" void *dlsym(void *handle, const char *symbol) {
    // std::cerr << "dlsym(" << symbol << ")" << std::endl << std::endl;

    // early out if not a CUDA driver symbol
    if (strncmp(symbol, "cu", 2) != 0) {
        return (real_dlsym(handle, symbol));
    }

    if (strcmp(symbol, "cuGetProcAddress") == 0) {
        return (void *)&cuGetProcAddress;
    }

    return (real_dlsym(handle, symbol));
}