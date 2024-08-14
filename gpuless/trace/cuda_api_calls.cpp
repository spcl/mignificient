#include "iostream"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <spdlog/spdlog.h>

#include "cuda_api_calls.hpp"
#include "dlsym_util.hpp"
#include "libgpuless.hpp"

#include "trace_executor_shmem_client.hpp"

namespace gpuless {

// FIXME: singleton
MemPoolRead readers;

std::string CudaRuntimeApiCall::nativeErrorToString(uint64_t err) {
    auto str = "[cudart] " +
               std::string(cudaGetErrorString(static_cast<cudaError_t>(err)));
    return str;
}

/*
 * cudaMalloc
 */
CudaMalloc::CudaMalloc(size_t size) : devPtr(nullptr), size(size) {}

uint64_t CudaMalloc::executeNative(CudaVirtualDevice &vdev) {
    static auto real =
        (decltype(&cudaMalloc<void>))real_dlsym(RTLD_NEXT, "cudaMalloc");
    return real(&this->devPtr, size);
}

flatbuffers::Offset<FBCudaApiCall>
CudaMalloc::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCudaMalloc(
        builder, reinterpret_cast<uint64_t>(this->devPtr), this->size);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudaMalloc, api_call.Union());
    return api_call_union;
}

CudaMalloc::CudaMalloc(const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCudaMalloc();
    this->devPtr = reinterpret_cast<void *>(c->dev_ptr());
    this->size = c->size();
}

/*
 * cudaMemcpyH2D
 */
CudaMemcpyH2D::CudaMemcpyH2D(void *dst, const void *src, size_t size)
    : dst(dst), src(src), size(size), shared_name("") {}

CudaMemcpyH2D::CudaMemcpyH2D(void *dst, const void *src, size_t size, std::string shared_name)
    : dst(dst), src(src), size(size), shared_name(shared_name) {}

uint64_t CudaMemcpyH2D::executeNative(CudaVirtualDevice &vdev) {
    static auto real =
        (decltype(&cudaMemcpy))real_dlsym(RTLD_NEXT, "cudaMemcpy");
    //auto s = std::chrono::high_resolution_clock::now();
    //auto val = real(this->dst, this->buffer.data(), this->size, cudaMemcpyHostToDevice);
    auto val = real(this->dst, this->buffer_ptr, this->size, cudaMemcpyHostToDevice);
    //auto val = real(this->dst, this->buffer_, this->size,
    //            cudaMemcpyHostToDevice);
    //auto e = std::chrono::high_resolution_clock::now();
    //auto d =
    //    std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
    //    1000000.0;
    //std::cerr << "h2d " << d << " " << size << std::endl;
    return val;
}

flatbuffers::Offset<FBCudaApiCall>
CudaMemcpyH2D::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {

    auto s = std::chrono::high_resolution_clock::now();
    flatbuffers::Offset<FBCudaMemcpyH2D> api_call;
    if(!this->shared_name.empty()) {
      //std::cerr << "shmem" << std::endl;
      api_call =
            CreateFBCudaMemcpyH2D(builder, reinterpret_cast<uint64_t>(this->dst),
                                reinterpret_cast<uint64_t>(this->src), this->size,
                                builder.CreateString(this->shared_name),
                                0);
    } else {
      api_call =
        CreateFBCudaMemcpyH2D(builder, reinterpret_cast<uint64_t>(this->dst),
                              reinterpret_cast<uint64_t>(this->src), this->size,
                              builder.CreateString(this->shared_name),
                              this->buffer.size());
    }
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudaMemcpyH2D, api_call.Union());
    //auto e = std::chrono::high_resolution_clock::now();
    //auto d =
    //    std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
    //    1000000.0;

    //std::cerr << "h2d ser " << d << std::endl;
    return api_call_union;
}


CudaMemcpyH2D::CudaMemcpyH2D(const FBCudaApiCall *fb_cuda_api_call) {
    //auto s = std::chrono::high_resolution_clock::now();
    auto c = fb_cuda_api_call->api_call_as_FBCudaMemcpyH2D();
    //auto e = std::chrono::high_resolution_clock::now();
    //auto d =
    //    std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
    //    1000000.0;
    //std::cerr << "h2d deser2 " << d << std::endl;
    this->dst = reinterpret_cast<void *>(c->dst());
    this->src = reinterpret_cast<void *>(c->src());
    this->size = c->size();
    this->buffer.resize(c->sent_bytes());
    //this->sent_bytes = 0;
    this->shared_name = *c->mmap()->c_str();
    //this->buffer = std::vector<uint8_t>(c->size());
    //std::memcpy(this->buffer.data(), c->buffer()->data(), c->buffer()->size());


    if(c->mmap()->size() == 0) {

      // Read from TCP?
      if(this->buffer.size() > 0) {
        recv_buffer(tcp_socket(), this->buffer, this->buffer.size());
      }

      //this->buffer_ptr = const_cast<unsigned char*>(c->buffer()->data());
    } else {

      //auto s = std::chrono::high_resolution_clock::now();
      auto ptr = readers.get(c->mmap()->c_str());
      //MemChunk chunk{nullptr, c->mmap()->c_str()};
      //chunk.open();
      //this->buffer_ptr = reinterpret_cast<unsigned char*>(chunk.ptr);
      this->buffer_ptr = reinterpret_cast<unsigned char*>(ptr);
      //auto e = std::chrono::high_resolution_clock::now();
      //auto d =
      //    std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
      //    1000000.0;
      //std::cerr << "h2d deser2 " << d << std::endl;
    }
    //this->buffer = const_cast<uint8_t*>(c->buffer()->data());
    //this->buffer_ = const_cast<uint8_t*>(c->buffer()->data());
    //this->buffer_ = const_cast<char*>(c->buffer()->data());
}

/*
 * cudaMemcpyD2H
 */
CudaMemcpyD2H::CudaMemcpyD2H(void *dst, const void *src, size_t size)
    : dst(dst), src(src), size(size), buffer(size), buffer_ptr(nullptr) { 

CudaMemcpyD2H::CudaMemcpyD2H(void *dst, const void *src, size_t size, std::string shared_name)
    : dst(dst), src(src), size(size), buffer_ptr(nullptr), shared_name(shared_name) {

uint64_t CudaMemcpyD2H::executeNative(CudaVirtualDevice &vdev) {
    static auto real =
        (decltype(&cudaMemcpy))real_dlsym(RTLD_NEXT, "cudaMemcpy");
    //return real(this->buffer.data(), this->src, this->size,
    //            cudaMemcpyDeviceToHost);
    return real(this->buffer_ptr, this->src, this->size,
                cudaMemcpyDeviceToHost);
}

flatbuffers::Offset<FBCudaApiCall>
CudaMemcpyD2H::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {
    //auto s = std::chrono::high_resolution_clock::now();
    flatbuffers::Offset<FBCudaMemcpyD2H> api_call;

    if(! this->shared_name.empty()) {
      api_call =
            CreateFBCudaMemcpyD2H(builder, reinterpret_cast<uint64_t>(this->dst),
                                reinterpret_cast<uint64_t>(this->src), this->size,
                                //builder.CreateVector(this->buffer));
                                builder.CreateString(this->shared_name),
                                builder.CreateVector(static_cast<uint8_t*>(nullptr), 0));
    } else if(buffer_ptr) {
      api_call =
            CreateFBCudaMemcpyD2H(builder, reinterpret_cast<uint64_t>(this->dst),
                                reinterpret_cast<uint64_t>(this->src), this->size,
                                //builder.CreateVector(this->buffer));
                                builder.CreateString(this->shared_name),
                                builder.CreateVector(this->buffer_ptr, this->size));
                                //builder.CreateString(this->buffer.data(), this->buffer.size()));
    } else {
      api_call =
          CreateFBCudaMemcpyD2H(builder, reinterpret_cast<uint64_t>(this->dst),
                                reinterpret_cast<uint64_t>(this->src), this->size,
                                //builder.CreateVector(this->buffer));
                                builder.CreateString(this->shared_name),
                                //builder.CreateVector(this->buffer));
                                builder.CreateVector(static_cast<uint8_t*>(nullptr), 0));
                                //builder.CreateString(this->buffer.data(), this->buffer.size()));
    }
    //auto e = std::chrono::high_resolution_clock::now();
    //auto d =
    //    std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
    //    1000000.0;
    //std::cerr << "d2h ser " << d << std::endl;
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudaMemcpyD2H, api_call.Union());
    return api_call_union;
}

CudaMemcpyD2H::CudaMemcpyD2H(const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCudaMemcpyD2H();
    this->dst = reinterpret_cast<void *>(c->dst());
    this->src = reinterpret_cast<void *>(c->src());
    this->size = c->size();
    // we might have received empty buffer, no data send -> allocate it!
    this->shared_name = *c->mmap()->c_str();

    if(c->mmap()->size() == 0) {
      //this->buffer_ptr = const_cast<unsigned char*>(c->buffer()->data());
      this->buffer_ptr = const_cast<unsigned char*>(c->buffer()->data());
      if(this->buffer.size() == 0) {
        this->buffer.resize(this->size);
      }
    } else {

      //auto s = std::chrono::high_resolution_clock::now();
      auto ptr = readers.get(c->mmap()->c_str());
      this->buffer_ptr = reinterpret_cast<unsigned char*>(ptr);
      //auto e = std::chrono::high_resolution_clock::now();
      //auto d =
      //    std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
      //    1000000.0;
      //std::cerr << "h2d deser2 " << d << std::endl;
    }
}

/*
 * cudaMemcpyD2D
 */
CudaMemcpyD2D::CudaMemcpyD2D(void *dst, const void *src, size_t size)
    : dst(dst), src(src), size(size) {}

uint64_t CudaMemcpyD2D::executeNative(CudaVirtualDevice &vdev) {
    static auto real =
        (decltype(&cudaMemcpy))real_dlsym(RTLD_NEXT, "cudaMemcpy");
    return real(this->dst, this->src, this->size, cudaMemcpyDeviceToDevice);
}

flatbuffers::Offset<FBCudaApiCall>
CudaMemcpyD2D::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCudaMemcpyD2D(
        builder, reinterpret_cast<uint64_t>(this->dst),
        reinterpret_cast<uint64_t>(this->src), this->size);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudaMemcpyD2D, api_call.Union());
    return api_call_union;
}

CudaMemcpyD2D::CudaMemcpyD2D(const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCudaMemcpyD2D();
    this->dst = reinterpret_cast<void *>(c->dst());
    this->src = reinterpret_cast<void *>(c->src());
    this->size = c->size();
}

/*
 * cudaMemcpyAsyncH2D
 */
CudaMemcpyAsyncH2D::CudaMemcpyAsyncH2D(void *dst, const void *src, size_t size,
                                       cudaStream_t stream)
    : dst(dst), src(src), size(size), stream(stream), buffer(size) {}

uint64_t CudaMemcpyAsyncH2D::executeNative(CudaVirtualDevice &vdev) {
    static auto real =
        (decltype(&cudaMemcpyAsync))real_dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    return real(this->dst, this->buffer.data(), this->size,
                cudaMemcpyHostToDevice, this->stream);
}

flatbuffers::Offset<FBCudaApiCall>
CudaMemcpyAsyncH2D::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCudaMemcpyAsyncH2D(
        builder, reinterpret_cast<uint64_t>(this->dst),
        reinterpret_cast<uint64_t>(this->src), this->size,
        reinterpret_cast<uint64_t>(this->stream),
        builder.CreateVector(this->buffer));
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudaMemcpyAsyncH2D, api_call.Union());
    return api_call_union;
}

CudaMemcpyAsyncH2D::CudaMemcpyAsyncH2D(const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCudaMemcpyAsyncH2D();
    this->dst = reinterpret_cast<void *>(c->dst());
    this->src = reinterpret_cast<void *>(c->src());
    this->size = c->size();
    this->stream = reinterpret_cast<cudaStream_t>(c->stream());
    this->buffer = std::vector<uint8_t>(c->size());
    std::memcpy(this->buffer.data(), c->buffer()->data(), c->buffer()->size());
}

/*
 * cudaMemcpyAsyncD2H
 */
CudaMemcpyAsyncD2H::CudaMemcpyAsyncD2H(void *dst, const void *src, size_t size,
                                       cudaStream_t stream)
    : dst(dst), src(src), size(size), stream(stream), buffer(size) {}

uint64_t CudaMemcpyAsyncD2H::executeNative(CudaVirtualDevice &vdev) {
    static auto real =
        (decltype(&cudaMemcpyAsync))real_dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    return real(this->buffer.data(), this->src, this->size,
                cudaMemcpyDeviceToHost, this->stream);
}

flatbuffers::Offset<FBCudaApiCall>
CudaMemcpyAsyncD2H::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCudaMemcpyAsyncD2H(
        builder, reinterpret_cast<uint64_t>(this->dst),
        reinterpret_cast<uint64_t>(this->src), this->size,
        reinterpret_cast<uint64_t>(this->stream),
        builder.CreateVector(this->buffer));
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudaMemcpyAsyncD2H, api_call.Union());
    return api_call_union;
}

CudaMemcpyAsyncD2H::CudaMemcpyAsyncD2H(const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCudaMemcpyAsyncD2H();
    this->dst = reinterpret_cast<void *>(c->dst());
    this->src = reinterpret_cast<void *>(c->src());
    this->size = c->size();
    this->stream = reinterpret_cast<cudaStream_t>(c->stream());
    this->buffer = std::vector<uint8_t>(c->size());
    std::memcpy(this->buffer.data(), c->buffer()->data(), c->buffer()->size());
}

/*
 * cudaMemcpyAsyncD2D
 */
CudaMemcpyAsyncD2D::CudaMemcpyAsyncD2D(void *dst, const void *src, size_t size,
                                       cudaStream_t stream)
    : dst(dst), src(src), size(size), stream(stream) {}

uint64_t CudaMemcpyAsyncD2D::executeNative(CudaVirtualDevice &vdev) {
    static auto real =
        (decltype(&cudaMemcpyAsync))real_dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    return real(this->dst, this->src, this->size, cudaMemcpyDeviceToDevice,
                this->stream);
}

flatbuffers::Offset<FBCudaApiCall>
CudaMemcpyAsyncD2D::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCudaMemcpyAsyncD2D(
        builder, reinterpret_cast<uint64_t>(this->dst),
        reinterpret_cast<uint64_t>(this->src), this->size,
        reinterpret_cast<uint64_t>(this->stream));
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudaMemcpyAsyncD2D, api_call.Union());
    return api_call_union;
}

CudaMemcpyAsyncD2D::CudaMemcpyAsyncD2D(const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCudaMemcpyAsyncD2D();
    this->dst = reinterpret_cast<void *>(c->dst());
    this->src = reinterpret_cast<void *>(c->src());
    this->size = c->size();
    this->stream = reinterpret_cast<cudaStream_t>(c->stream());
}

/*
 * cudaFree
 */
CudaFree::CudaFree(void *devPtr) : devPtr(devPtr) {}

uint64_t CudaFree::executeNative(CudaVirtualDevice &vdev) {
    static auto real = (decltype(&cudaFree))real_dlsym(RTLD_NEXT, "cudaFree");
    return real(this->devPtr);
}

flatbuffers::Offset<FBCudaApiCall>
CudaFree::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {
    auto api_call =
        CreateFBCudaFree(builder, reinterpret_cast<uint64_t>(this->devPtr));
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudaFree, api_call.Union());
    return api_call_union;
}

CudaFree::CudaFree(const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCudaFree();
    this->devPtr = reinterpret_cast<void *>(c->dev_ptr());
}

/*
 * cudaLaunchKernel
 */
CudaLaunchKernel::CudaLaunchKernel(
    std::string symbol, std::vector<uint64_t> required_cuda_modules,
    std::vector<std::string> required_function_symbols, const void *fnPtr,
    const dim3 &gridDim, const dim3 &blockDim, size_t sharedMem,
    cudaStream_t stream, std::vector<std::vector<uint8_t>> &paramBuffers,
    std::vector<KParamInfo> &paramInfos)
    : symbol(symbol), required_cuda_modules_(required_cuda_modules),
      required_function_symbols_(required_function_symbols), fnPtr(fnPtr),
      gridDim(gridDim), blockDim(blockDim), sharedMem(sharedMem),
      stream(stream), paramBuffers(paramBuffers), paramInfos(paramInfos) {}

uint64_t CudaLaunchKernel::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cuLaunchKernel);

    auto fn_reg_it = vdev.function_registry_.find(this->symbol);
    if (fn_reg_it == vdev.function_registry_.end()) {
        SPDLOG_ERROR("Function not registered: {}", this->symbol);
        std::exit(EXIT_FAILURE);
    }

    std::vector<void *> args;
    for (unsigned i = 0; i < paramBuffers.size(); i++) {
        auto &b = this->paramBuffers[i];

        args.push_back(b.data());
    }

    auto ret = real(fn_reg_it->second, this->gridDim.x, this->gridDim.y,
                    this->gridDim.z, this->blockDim.x, this->blockDim.y,
                    this->blockDim.z, this->sharedMem, this->stream,
                    args.data(), nullptr);

    if (ret != CUDA_SUCCESS) {
        const char *err_str;
        cuGetErrorString(ret, &err_str);
        SPDLOG_ERROR("cuLaunchKernel() failed: {}", err_str);
    }

    return cudaSuccess;
}

std::vector<uint64_t> CudaLaunchKernel::requiredCudaModuleIds() {
    return this->required_cuda_modules_;
}

std::vector<std::string> CudaLaunchKernel::requiredFunctionSymbols() {
    return this->required_function_symbols_;
}

flatbuffers::Offset<FBCudaApiCall>
CudaLaunchKernel::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {
    std::vector<flatbuffers::Offset<flatbuffers::String>> fb_req_fns;
    for (const auto &f : this->requiredFunctionSymbols()) {
        fb_req_fns.push_back(builder.CreateString(f));
    }
    std::vector<flatbuffers::Offset<FBParamBuffer>> fb_param_buffers;
    for (const auto &p : this->paramBuffers) {
        fb_param_buffers.push_back(
            CreateFBParamBuffer(builder, builder.CreateVector(p)));
    }
    std::vector<flatbuffers::Offset<FBParamInfo>> fb_param_infos;
    for (const auto &p : this->paramInfos) {
        fb_param_infos.push_back(
            CreateFBParamInfo(builder, builder.CreateString(p.paramName),
                              static_cast<FBPtxParameterType>(p.type),
                              p.typeSize, p.align, p.size));
    }

    auto gdim = FBDim3{this->gridDim.x, this->gridDim.y, this->gridDim.z};
    auto bdim = FBDim3{this->blockDim.x, this->blockDim.y, this->blockDim.z};

    auto api_call = CreateFBCudaLaunchKernel(
        builder, builder.CreateString(this->symbol),
        builder.CreateVector(this->requiredCudaModuleIds()),
        builder.CreateVector(fb_req_fns), &gdim, &bdim, this->sharedMem,
        reinterpret_cast<uint64_t>(this->stream),
        builder.CreateVector(fb_param_buffers),
        builder.CreateVector(fb_param_infos));

    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudaLaunchKernel, api_call.Union());
    return api_call_union;
}

CudaLaunchKernel::CudaLaunchKernel(const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCudaLaunchKernel();
    const FBDim3 *fb_grid_dim = c->grid_dim();
    const FBDim3 *fb_block_dim = c->block_dim();
    const dim3 grid_dim{static_cast<unsigned int>(fb_grid_dim->x()),
                        static_cast<unsigned int>(fb_grid_dim->y()),
                        static_cast<unsigned int>(fb_grid_dim->z())};
    const dim3 block_dim{static_cast<unsigned int>(fb_block_dim->x()),
                         static_cast<unsigned int>(fb_block_dim->y()),
                         static_cast<unsigned int>(fb_block_dim->z())};
    auto stream_ = reinterpret_cast<cudaStream_t>(c->stream());

    std::vector<std::vector<uint8_t>> pb;
    for (const auto &b : *c->param_buffers()) {
        pb.emplace_back(b->buffer()->size());
        std::memcpy(pb.back().data(), b->buffer()->data(), b->buffer()->size());
    }

    std::vector<KParamInfo> kpi;
    for (const auto &i : *c->param_infos()) {
        KParamInfo info{i->name()->str(),
                        static_cast<PtxParameterType>(i->ptx_param_type()),
                        static_cast<int>(i->type_size()),
                        static_cast<int>(i->align()),
                        static_cast<int>(i->size())};
        kpi.push_back(info);
    }

    this->symbol = c->symbol()->str();

    // required modules and functions are shipped outside this object
    this->required_cuda_modules_ = std::vector<uint64_t>();
    this->required_function_symbols_ = std::vector<std::string>();

    this->fnPtr = nullptr;
    this->gridDim = grid_dim;
    this->blockDim = block_dim;
    this->sharedMem = c->shared_mem();
    this->stream = stream_;
    this->paramBuffers = pb;
    this->paramInfos = kpi;
}

/*
 * cudaStreamSynchronize
 */
CudaStreamSynchronize::CudaStreamSynchronize(cudaStream_t stream)
    : stream(stream) {}

uint64_t CudaStreamSynchronize::executeNative(CudaVirtualDevice &vdev) {
    static auto real = (decltype(&cudaStreamSynchronize))real_dlsym(
        RTLD_NEXT, "cudaStreamSynchronize");
    return real(this->stream);
}

flatbuffers::Offset<FBCudaApiCall>
CudaStreamSynchronize::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCudaStreamSynchronize(
        builder, reinterpret_cast<uint64_t>(this->stream));
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudaStreamSynchronize, api_call.Union());
    return api_call_union;
}

CudaStreamSynchronize::CudaStreamSynchronize(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCudaStreamSynchronize();
    this->stream = reinterpret_cast<cudaStream_t>(c->stream());
}

/*
 * cudaGetDeviceProperties
 */
CudaGetDeviceProperties::CudaGetDeviceProperties() = default;

uint64_t CudaGetDeviceProperties::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cudaGetDeviceProperties);
    return real(&this->properties, 0);
}

CudaGetDeviceProperties::CudaGetDeviceProperties(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCudaGetDeviceProperties();
    std::memcpy(&this->properties, c->properties_data()->data(),
                sizeof(cudaDeviceProp));
}

flatbuffers::Offset<FBCudaApiCall>
CudaGetDeviceProperties::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {
    std::vector<uint8_t> properites_data(sizeof(cudaDeviceProp));
    std::memcpy(properites_data.data(), &this->properties,
                sizeof(cudaDeviceProp));
    auto api_call = CreateFBCudaGetDeviceProperties(
        builder, builder.CreateVector(properites_data));
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudaGetDeviceProperties,
        api_call.Union());
    return api_call_union;
}

/*
 * cudaDeviceSynchronize
 */
CudaDeviceSynchronize::CudaDeviceSynchronize() = default;

CudaDeviceSynchronize::CudaDeviceSynchronize(
    const FBCudaApiCall *fb_cuda_api_call) {}

uint64_t CudaDeviceSynchronize::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cudaDeviceSynchronize);
    return real();
}

flatbuffers::Offset<FBCudaApiCall>
CudaDeviceSynchronize::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCudaDeviceSynchronize(builder);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudaDeviceSynchronize, api_call.Union());
    return api_call_union;
}

/*
 * cudaFuncGetAttributes
 */
CudaFuncGetAttributes::CudaFuncGetAttributes(
    std::string symbol, std::vector<uint64_t> requiredCudaModules,
    std::vector<std::string> requiredFunctionSymbols)
    : symbol(std::move(symbol)),
      required_cuda_modules_(std::move(requiredCudaModules)),
      required_function_symbols_(std::move(requiredFunctionSymbols)) {}

CudaFuncGetAttributes::CudaFuncGetAttributes(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCudaFuncGetAttributes();

    this->cfa.binaryVersion = c->binary_version();
    this->cfa.cacheModeCA = c->cache_mode_ca();
    this->cfa.constSizeBytes = c->const_size_bytes();
    this->cfa.localSizeBytes = c->local_size_bytes();
    this->cfa.maxDynamicSharedSizeBytes = c->max_dynamic_shared_size_bytes();
    this->cfa.maxThreadsPerBlock = c->max_threads_per_block();
    this->cfa.numRegs = c->num_regs();
    this->cfa.preferredShmemCarveout = c->preferred_shmem_carveout();
    this->cfa.ptxVersion = c->ptx_version();
    this->cfa.sharedSizeBytes = c->shared_size_bytes();
    this->symbol = c->symbol()->str();
}

uint64_t CudaFuncGetAttributes::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cuFuncGetAttribute);

    auto fn_reg_it = vdev.function_registry_.find(this->symbol);
    if (fn_reg_it == vdev.function_registry_.end()) {
        SPDLOG_ERROR("Function not registered: {}", this->symbol);
        std::exit(EXIT_FAILURE);
    }

    CUfunction f = fn_reg_it->second;

    real(&this->cfa.binaryVersion, CU_FUNC_ATTRIBUTE_BINARY_VERSION, f);
    real(&this->cfa.cacheModeCA, CU_FUNC_ATTRIBUTE_CACHE_MODE_CA, f);
    real(reinterpret_cast<int *>(&this->cfa.constSizeBytes),
         CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, f);
    real(reinterpret_cast<int *>(&this->cfa.localSizeBytes),
         CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, f);
    real(&this->cfa.maxDynamicSharedSizeBytes,
         CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, f);
    real(&this->cfa.maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
         f);
    real(&this->cfa.numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, f);
    real(&this->cfa.preferredShmemCarveout,
         CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, f);
    real(&this->cfa.ptxVersion, CU_FUNC_ATTRIBUTE_PTX_VERSION, f);
    real(reinterpret_cast<int *>(&this->cfa.sharedSizeBytes),
         CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, f);

    return cudaSuccess;
}

flatbuffers::Offset<FBCudaApiCall>
CudaFuncGetAttributes::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCudaFuncGetAttributes(
        builder, builder.CreateString(this->symbol), this->cfa.binaryVersion,
        this->cfa.cacheModeCA, this->cfa.constSizeBytes,
        this->cfa.localSizeBytes, this->cfa.maxDynamicSharedSizeBytes,
        this->cfa.maxThreadsPerBlock, this->cfa.numRegs,
        this->cfa.preferredShmemCarveout, this->cfa.ptxVersion,
        this->cfa.sharedSizeBytes);

    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCudaFuncGetAttributes, api_call.Union());
    return api_call_union;
}

std::vector<uint64_t> CudaFuncGetAttributes::requiredCudaModuleIds() {
    return this->required_cuda_modules_;
}

std::vector<std::string> CudaFuncGetAttributes::requiredFunctionSymbols() {
    return this->required_function_symbols_;
}

} // namespace gpuless
