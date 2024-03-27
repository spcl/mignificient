#include <chrono>
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <cudnn.h>

#include "../trace/cubin_analysis.hpp"
#include "../utils.hpp"

#define LINK_CU_FUNCTION(symbol, f)                                            \
    do {                                                                       \
        if (strcmp(symbol, #f) == 0) {                                         \
            *pfn = (void *)&f;                                                 \
            return CUDA_SUCCESS;                                               \
        }                                                                      \
    } while (0)

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
    static auto internal_dlsym = (decltype(&dlsym))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
    return (*internal_dlsym)(handle, symbol);
}

std::map<void *, std::string> &getCUfnptrToSymbolMap() {
    static std::map<void *, std::string> cufnptr_to_symbol_map;
    return cufnptr_to_symbol_map;
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

extern "C" cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size) {
    using T = cudaError_t (*)(void **, size_t);
    static auto real_func = (T)real_dlsym(RTLD_NEXT, __func__);
    cudaError_t err = real_func(devPtr, size);
    std::cerr << "cudaMalloc(" << addrToIdentDevice(*devPtr) << " (" << std::hex
              << *devPtr << "), " << size << " bytes)" << std::endl
              << std::endl;
    return err;
}

extern "C" cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size) {
    using T = cudaError_t (*)(void **, size_t);
    static auto real_func = (T)real_dlsym(RTLD_NEXT, __func__);
    std::cerr << "cudaMallocHost()" << std::endl << std::endl;
    return real_func(ptr, size);
}

extern "C" cudaError_t CUDARTAPI cudaMallocAsync(void **devPtr, size_t size,
                                                 cudaStream_t hStream) {
    using T = cudaError_t (*)(void **, size_t, cudaStream_t);
    static auto real_func = (T)real_dlsym(RTLD_NEXT, __func__);
    cudaError_t err = real_func(devPtr, size, hStream);
    std::cerr << "cudaMallocAsync(" << addrToIdentDevice(*devPtr) << " ("
              << std::hex << *devPtr << "), " << size << " bytes"
              << ", stream " << ((uint64_t)hStream) << ")" << std::endl
              << std::endl;
    return err;
}

extern "C" void CUDARTAPI __cudaRegisterFunction(
    void **fatCubinHandle, const char *hostFun, char *deviceFun,
    const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid,
    dim3 *bDim, dim3 *gDim, int *wSize) {
    static auto real_func =
        (decltype(&__cudaRegisterFunction))real_dlsym(RTLD_NEXT, __func__);
//    std::cerr << "__cudaRegisterFunction(" << deviceName << ")"
//        << std::endl;
    auto &map = getCUfnptrToSymbolMap();
    map.emplace(std::make_pair((void *)deviceFun, deviceName));
    map.emplace(std::make_pair((void *)hostFun, deviceName));
    real_func(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid,
              bid, bDim, gDim, wSize);
}

extern "C" cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src,
                                                 size_t count,
                                                 enum cudaMemcpyKind kind,
                                                 cudaStream_t stream) {
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

    auto s = std::chrono::high_resolution_clock::now();

    //    float probe[4];
    //    static auto real_memcpy =
    //        (decltype(&cudaMemcpy))real_dlsym(RTLD_NEXT, "cudaMemcpy");
    //    if (kind == cudaMemcpyKind::cudaMemcpyDeviceToDevice) {
    //        real_memcpy(probe, dst, 4 * sizeof(float),
    //        cudaMemcpyDeviceToHost); fprintf(stderr,
    //                "cudaMemcpyAsyncD2D probe (y=%p): y[0]=%f, y[1]=%f,
    //                y[2]=%f, " "y[3]=%f\n\n", dst, probe[0], probe[1],
    //                probe[2], probe[3]);
    //    }

    cudaError_t err = real_func(dst, src, count, kind, stream);

    //    if (kind == cudaMemcpyKind::cudaMemcpyDeviceToDevice) {
    //        real_memcpy(probe, dst, 4 * sizeof(float),
    //        cudaMemcpyDeviceToHost); fprintf(stderr,
    //                "cudaMemcpyAsyncD2D probe (y=%p): y[0]=%f, y[1]=%f,
    //                y[2]=%f, " "y[3]=%f\n\n", dst, probe[0], probe[1],
    //                probe[2], probe[3]);
    //    }

    auto e = std::chrono::high_resolution_clock::now();
    auto d =
        std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
        1000.0;
    acc_time += d;

    ss << " [time: " << d << " ms, acc_time: " << acc_time << " ms]"
       << std::endl;

//    if (kind == cudaMemcpyKind::cudaMemcpyDeviceToHost) {
//        size_t nbytes = std::min(count, 16UL);
//        auto *dst_byte_ptr = static_cast<uint8_t *>(dst);
//
//        ss << "D2H memory probe: ";
//        for (size_t i = 0; i < nbytes; i++) {
//            ss << std::hex << std::setfill('0') << std::setw(2)
//               << (int)dst_byte_ptr[i] << " ";
//        }
//        ss << std::endl;
//    }

    std::cerr << ss.str() << std::endl;

    return err;
}

extern "C" cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src,
                                            size_t count,
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

    auto s = std::chrono::high_resolution_clock::now();
    cudaError_t err = real_func(dst, src, count, kind);
    auto e = std::chrono::high_resolution_clock::now();
    auto d =
        std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
        1000.0;
    acc_time += d;

    ss << " [time: " << d << " ms, acc_time: " << acc_time << " ms]"
       << std::endl;
    std::cerr << ss.str() << std::endl;

    return err;
}

// extern "C" cudaError_t CUDARTAPI cudaLaunchKernel(const void *func,
//                                                   dim3 gridDim, dim3
//                                                   blockDim, void **args,
//                                                   size_t sharedMem,
//                                                   cudaStream_t stream) {
//     static fnCudaLaunchKernel real_func =
//         (fnCudaLaunchKernel)real_dlsym(RTLD_NEXT, __func__);

//     cudaError_t err;
//     auto analyzer = getAnalyzer();

//     // get kernel parameters
//     auto &map = getCUfnptrToSymbolMap();
//     auto it = map.find((void *)func);
//     if (it != map.end()) {
//         auto s = std::chrono::high_resolution_clock::now();
//         err = real_func(func, gridDim, blockDim, args, sharedMem, stream);
//         auto e = std::chrono::high_resolution_clock::now();
//         auto d = std::chrono::duration_cast<std::chrono::microseconds>(e - s)
//                      .count() /
//                  1000.0;

//         std::string &kernel_symbol = it->second;
//         std::vector<KParamInfo> params;
//         analyzer.kernel_parameters(kernel_symbol, params);

//         std::stringstream ss;

//         // kernel symbol name
//         std::string cmd = "echo " + kernel_symbol + "| c++filt";
//         auto demangled_symbol = exec(cmd.c_str());
//         ss << demangled_symbol;

//         // grid/block configuaration
//         ss << "grid(";
//         ss << gridDim.x << ",";
//         ss << gridDim.y << ",";
//         ss << gridDim.z;
//         ss << "),block(";
//         ss << blockDim.x << ",";
//         ss << blockDim.y << ",";
//         ss << blockDim.z;
//         ss << ")" << std::endl;

//         // stream
//         ss << "sharedMem: " << sharedMem << std::endl;
//         ss << "stream: " << stream << std::endl;

//         // parameters
//         ss << "args ( ";
//         for (int i = 0; i < params.size(); i++) {
//             const auto &p = params[i];
//             void *d_ptr = *((void **)args[i]);
//             if (!MAP_PTR_TO_IDENT) { // print as memory address
//                 ss << "0x" << std::hex << d_ptr;
//             } else { // print as identifier
//                 ss << addrToIdentDevice(d_ptr) << "(" << std::hex << d_ptr
//                    << ")";
//             }

//             if (i < params.size() - 1) {
//                 ss << ", ";
//             }
//         }
//         ss << " )" << std::endl;

//         // execution time
//         ss << "time: " << d << " ms" << std::endl << std::endl;

//         std::cerr << ss.str();
//     } else {
//         std::cerr << "[error] no symbol found" << std::endl;
//     }

//     return err;
// }

extern "C" cudaError_t CUDARTAPI
cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags) {
    static auto real_func =
        (decltype(&cudaStreamCreateWithFlags))real_dlsym(RTLD_NEXT, __func__);
    std::cerr << "cudaStreamCreateWithFlags()" << std::endl;
    return real_func(pStream, flags);
}

extern "C" cudaError_t
cudaStreamIsCapturing(cudaStream_t stream,
                      enum cudaStreamCaptureStatus *pCaptureStatus) {
    static auto real_func =
        (decltype(&cudaStreamIsCapturing))real_dlsym(RTLD_NEXT, __func__);

    auto s = std::chrono::high_resolution_clock::now();
    auto err = real_func(stream, pCaptureStatus);
    auto e = std::chrono::high_resolution_clock::now();
    auto d =
        std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
        1000.0;
    acc_time += d;

    std::cerr << "cudaStreamIsCapturing()";
    std::cerr << " [time: " << d << " ms, acc_time: " << acc_time << " ms]"
              << std::endl
              << std::endl;

    return err;
}

extern "C" cudaError_t cudaStreamBeginCapture(cudaStream_t stream,
                                              cudaStreamCaptureMode mode) {
    static auto real_func =
        (decltype(&cudaStreamBeginCapture))real_dlsym(RTLD_NEXT, __func__);
    std::cerr << "cudaStreamBeginCapture()" << std::endl << std::endl;
    return real_func(stream, mode);
}

extern "C" cudaError_t cudaStreamEndCapture(cudaStream_t stream,
                                            cudaGraph_t *pGraph) {
    static auto real_func =
        (decltype(&cudaStreamEndCapture))real_dlsym(RTLD_NEXT, __func__);
    std::cerr << "cudaStreamEndCapture()" << std::endl << std::endl;
    return real_func(stream, pGraph);
}

extern "C" CUresult CUDAAPI cuLaunchKernel(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams, void **extra) {
    static auto real_func =
        (decltype(&cuLaunchKernel))real_dlsym(RTLD_NEXT, __func__);

    std::stringstream ss;
    ss << "cuLaunchKernel(";

    auto &map = getCUfnptrToSymbolMap();
    auto it = map.find((void *)f);
    if (it != map.end()) {
        std::string &kernel_symbol = it->second;
        std::string cmd = "echo " + kernel_symbol + "| c++filt";
        auto demangled_symbol = exec(cmd.c_str());
        ss << string_rstrip(demangled_symbol);
        // ss << demangled_symbol;
    } else {
        ss << "unknown kernel";
    }

    ss << ")";

    auto s = std::chrono::high_resolution_clock::now();
    auto err =
        real_func(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                  blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
    auto e = std::chrono::high_resolution_clock::now();
    auto d =
        std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
        1000.0;
    acc_time += d;

    ss << " [time: " << d << " ms, acc_time: " << acc_time << " ms]"
       << std::endl;
    std::cerr << ss.str() << std::endl;

    return err;
}

extern "C" void **CUDARTAPI __cudaRegisterFatBinary(void *fatCubin) {
    static auto real_func =
        (decltype(&__cudaRegisterFatBinary))real_dlsym(RTLD_NEXT, __func__);
//    std::cerr << "__cudaRegisterFatBinary()" << std::endl;
    return real_func(fatCubin);
}

extern "C" CUresult cuModuleLoad(CUmodule *module, const char *fname) {
    static auto real_func =
        (decltype(&cuModuleLoad))real_dlsym(RTLD_NEXT, __func__);
    //    std::cerr << "cuModuleLoad()" << std::endl;
    return real_func(module, fname);
}

extern "C" CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    static auto real_func =
        (decltype(&cuModuleLoadData))real_dlsym(RTLD_NEXT, __func__);
    //    std::cerr << "cuModuleLoadData()" << std::endl;
    return real_func(module, image);
}

extern "C" CUresult cuModuleLoadDataEx(CUmodule *module, const void *image,
                                       unsigned int numOptions,
                                       CUjit_option *options,
                                       void **optionValues) {
    static auto real_func =
        (decltype(&cuModuleLoadDataEx))real_dlsym(RTLD_NEXT, __func__);
    std::cerr << "cuModuleLoadDataEx()" << std::endl;
    return real_func(module, image, numOptions, options, optionValues);
}

extern "C" CUresult cuModuleLoadFatBinary(CUmodule *module,
                                          const void *fatCubin) {
    static auto real_func =
        (decltype(&cuModuleLoadFatBinary))real_dlsym(RTLD_NEXT, __func__);
    //    std::cerr << "cuModuleLoadFatBinary()" << std::endl;
    return real_func(module, fatCubin);
}

extern "C" CUresult cuModuleUnload(CUmodule hmod) {
    static auto real_func =
        (decltype(&cuModuleUnload))real_dlsym(RTLD_NEXT, __func__);
    //    std::cerr << "cuModuleUnload()" << std::endl;
    return real_func(hmod);
}

extern "C" CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes,
                                      CUmodule hmod, const char *name) {
    static auto real_func =
        (decltype(&cuModuleGetGlobal))real_dlsym(RTLD_NEXT, __func__);
    //    std::cerr << "cuModuleGetGlobal(" << name << ")" << std::endl;
    return real_func(dptr, bytes, hmod, name);
}

extern "C" void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                                  char *deviceAddress, const char *deviceName,
                                  int ext, size_t size, int constant,
                                  int global) {
    static auto real_func =
        (decltype(&__cudaRegisterVar))real_dlsym(RTLD_NEXT, __func__);
    //    std::cerr << "__cudaRegisterVar(" << deviceName << ")" << std::endl;
    real_func(fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size,
              constant, global);
}

extern "C" CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
                                        const char *name) {
    static auto real_func =
        (decltype(&cuModuleGetFunction))real_dlsym(RTLD_NEXT, __func__);
    auto err = real_func(hfunc, hmod, name);
    auto &map = getCUfnptrToSymbolMap();
    map.emplace(std::make_pair((void *)*hfunc, name));
    return err;
}

extern "C" CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    static auto real_func =
        (decltype(&cuMemAlloc_v2))real_dlsym(RTLD_NEXT, __func__);
    std::cerr << "cuMemAlloc_v2" << std::endl << std::endl;
    return real_func(dptr, bytesize);
}

extern "C" CUresult CUDAAPI cuMemcpyDtoH_v2(void *dstHost,
                                            CUdeviceptr srcDevice,
                                            size_t ByteCount) {
    static auto real_func =
        (decltype(&cuMemcpyDtoH_v2))real_dlsym(RTLD_NEXT, __func__);
    std::cerr << "cuMemcpyDtoH_v2" << std::endl << std::endl;
    return real_func(dstHost, srcDevice, ByteCount);
}

extern "C" CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn,
                                             int cudaVersion,
                                             cuuint64_t flags) {
    static auto real_func =
        (decltype(&cuGetProcAddress))real_dlsym(RTLD_NEXT, __func__);
    // std::cerr << "cuGetProcAddress(" << symbol << ")" << std::endl <<
    // std::endl;

    LINK_CU_FUNCTION(symbol, cuLaunchKernel);
    LINK_CU_FUNCTION(symbol, cuModuleGetFunction);
    LINK_CU_FUNCTION(symbol, cuGetProcAddress);
    LINK_CU_FUNCTION(symbol, cuMemAlloc_v2);
    LINK_CU_FUNCTION(symbol, cuMemcpyDtoH_v2);
    LINK_CU_FUNCTION(symbol, cuModuleLoad);
    LINK_CU_FUNCTION(symbol, cuModuleLoadData);
    LINK_CU_FUNCTION(symbol, cuModuleLoadDataEx);
    LINK_CU_FUNCTION(symbol, cuModuleLoadFatBinary);
    LINK_CU_FUNCTION(symbol, cuModuleUnload);
    LINK_CU_FUNCTION(symbol, cuModuleGetGlobal);

    return real_func(symbol, pfn, cudaVersion, flags);
}

extern "C" cudnnStatus_t
cudnnConvolutionForward(cudnnHandle_t handle, const void *alpha,
                        const cudnnTensorDescriptor_t xDesc, const void *x,
                        const cudnnFilterDescriptor_t wDesc, const void *w,
                        const cudnnConvolutionDescriptor_t convDesc,
                        cudnnConvolutionFwdAlgo_t algo, void *workSpace,
                        size_t workSpaceSizeInBytes, const void *beta,
                        const cudnnTensorDescriptor_t yDesc, void *y) {
    static auto real_func =
        (decltype(&cudnnConvolutionForward))real_dlsym(RTLD_NEXT, __func__);

    //    static auto real_memcpy =
    //        (decltype(&cudaMemcpy))real_dlsym(RTLD_NEXT, "cudaMemcpy");
    //
    //    float probe[4];
    //    std::cerr << "cudnnConvolutionForward()" << std::endl << std::endl;
    //
    //    real_memcpy(probe, y, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    //    fprintf(stderr,
    //            "cudnnConvolutionForward probe (y=%p): y[0]=%f, y[1]=%f,
    //            y[2]=%f, " "y[3]=%f\n\n", y, probe[0], probe[1], probe[2],
    //            probe[3]);

    auto err = real_func(handle, alpha, xDesc, x, wDesc, w, convDesc, algo,
                         workSpace, workSpaceSizeInBytes, beta, yDesc, y);
    //
    //    real_memcpy(probe, y, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    //    fprintf(stderr,
    //            "cudnnConvolutionForward probe (y=%p): y[0]=%f, y[1]=%f,
    //            y[2]=%f, " "y[3]=%f\n\n", y, probe[0], probe[1], probe[2],
    //            probe[3]);

    return err;
}

extern "C" cudnnStatus_t cudnnCreate(cudnnHandle_t *handle) {
    std::cerr << "cudnnCreate()" << std::endl << std::endl;
    static auto real_func =
        (decltype(&cudnnCreate))real_dlsym(RTLD_NEXT, __func__);
    return real_func(handle);
}

extern "C" cudnnStatus_t
cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc) {
    std::cerr << "cudnnCreateTensorDescriptor()" << std::endl << std::endl;
    static auto real_func =
        (decltype(&cudnnCreateTensorDescriptor))real_dlsym(RTLD_NEXT, __func__);
    return real_func(tensorDesc);
}

extern "C" cudnnStatus_t
cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t derivedBnDesc,
                              const cudnnTensorDescriptor_t xDesc,
                              cudnnBatchNormMode_t mode) {
    std::cerr << "cudnnDeriveBNTensorDescriptor()" << std::endl << std::endl;
    static auto real_func =
        (decltype(&cudnnDeriveBNTensorDescriptor))real_dlsym(RTLD_NEXT,
                                                             __func__);
    return real_func(derivedBnDesc, xDesc, mode);
}

extern "C" cudnnStatus_t
cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc,
                           cudnnDataType_t dataType, int nbDims,
                           const int dimA[], const int strideA[]) {
    static auto real_func =
        (decltype(&cudnnSetTensorNdDescriptor))real_dlsym(RTLD_NEXT, __func__);
    std::cerr << "cudnnSetTensorNdDescriptor() [nbDims=" << nbDims << "]"
              << std::endl
              << std::endl;
    return real_func(tensorDesc, dataType, nbDims, dimA, strideA);
}

extern "C" cudnnStatus_t cudnnBatchNormalizationForwardInference(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha,
    const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, const void *estimatedMean,
    const void *estimatedVariance, double epsilon) {
    static auto real_func =
        (decltype(&cudnnBatchNormalizationForwardInference))real_dlsym(
            RTLD_NEXT, __func__);
    std::cerr << "cudnnBatchNormalizationForwardInference()" << std::endl
              << std::endl;
    return real_func(handle, mode, alpha, beta, xDesc, x, yDesc, y,
                     bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean,
                     estimatedVariance, epsilon);
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
