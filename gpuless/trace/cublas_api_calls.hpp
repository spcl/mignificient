#ifndef GPULESS_CUBLAS_API_CALLS_HPP
#define GPULESS_CUBLAS_API_CALLS_HPP

#include "cuda_api_calls.hpp"

namespace gpuless {

class CublasApiCAll : public AbstractCudaApiCall {
  public:
    std::string nativeErrorToString(uint64_t err) override;
};

class CublasCreateV2 : public CublasApiCAll {
  public:
    uint64_t virtual_handle;

    explicit CublasCreateV2(uint64_t virtualHandle);
    explicit CublasCreateV2(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CublasSetStreamV2 : public CublasApiCAll {
  public:
    uint64_t virtual_handle;
    cudaStream_t stream;

    CublasSetStreamV2(uint64_t virtualHandle, cudaStream_t stream);
    explicit CublasSetStreamV2(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CublasSetMathMode : public CublasApiCAll {
  public:
    uint64_t virtual_handle;
    cublasMath_t mode;

    CublasSetMathMode(uint64_t virtualHandle, cublasMath_t mode);
    explicit CublasSetMathMode(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CublasSgemmV2 : public CublasApiCAll {
  public:
    uint64_t virtual_handle;
    cublasOperation_t transa;
    cublasOperation_t transb;
    int m;
    int n;
    int k;
    float alpha;
    float beta;
    const float *A;
    const float *B;
    const float *C;
    int lda;
    int ldb;
    int ldc;

    CublasSgemmV2(uint64_t virtualHandle, cublasOperation_t transa,
                  cublasOperation_t transb, int m, int n, int k, float alpha,
                  float beta, const float *a, const float *b, const float *c,
                  int lda, int ldb, int ldc);
    explicit CublasSgemmV2(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CublasLtCreate : public CublasApiCAll {
  public:
    uint64_t virtual_handle;

    explicit CublasLtCreate(uint64_t virtualHandle);
    explicit CublasLtCreate(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;
    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

// class CublasLtCtxInit : public CublasApiCAll {
//   public:
//     uint64_t executeNative(CudaVirtualDevice &vdev) override;
//     flatbuffers::Offset<FBCudaApiCall>
//     fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
// };

class CublasLtMatmulDescCreate : public CublasApiCAll {
  public:
    uint64_t virtual_mmd;
    cublasComputeType_t compute_type;
    cudaDataType_t scale_type;
    CublasLtMatmulDescCreate(uint64_t virtualMmd,
                             cublasComputeType_t computeType,
                             cudaDataType_t scaleType);
    explicit CublasLtMatmulDescCreate(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;
    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CublasLtMatmulDescDestroy : public CublasApiCAll {
  public:
    uint64_t virtual_mmd;

    explicit CublasLtMatmulDescDestroy(uint64_t virtualMmd);
    explicit CublasLtMatmulDescDestroy(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;
    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CublasLtMatmulDescSetAttribute : public CublasApiCAll {
  public:
    uint64_t virtual_mmd;
    cublasLtMatmulDescAttributes_t attr;
    std::vector<uint8_t> buf;

    CublasLtMatmulDescSetAttribute(uint64_t virtualMmd,
                                   cublasLtMatmulDescAttributes_t attr,
                                   std::vector<uint8_t> buf);
    explicit CublasLtMatmulDescSetAttribute(
        const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;
    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CublasLtMatmul : public CublasApiCAll {
  public:
    uint64_t virtual_handle;
    uint64_t virtual_mmd;
    std::vector<uint8_t> alpha;
    std::vector<uint8_t> beta;
    const void *A;
    const void *B;
    const void *C;
    void *D;
    uint64_t virtual_ml_a_desc;
    uint64_t virtual_ml_b_desc;
    uint64_t virtual_ml_c_desc;
    uint64_t virtual_ml_d_desc;
    cublasLtMatmulAlgo_t algo{};
    bool algo_is_null;
    void *workspace;
    size_t workspace_size_in_bytes;
    cudaStream_t stream;

    CublasLtMatmul(uint64_t virtualHandle, uint64_t virtualMmd,
                   std::vector<uint8_t> &alpha, std::vector<uint8_t> &beta,
                   const void *a, const void *b, const void *c, void *d,
                   uint64_t virtualMlADesc, uint64_t virtualMlBDesc,
                   uint64_t virtualMlCDesc, uint64_t virtualMlDDesc,
                   const cublasLtMatmulAlgo_t &algo, bool algoIsNull,
                   void *workspace, size_t workspaceSizeInBytes,
                   cudaStream_t stream);
    explicit CublasLtMatmul(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;
    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CublasLtMatrixLayoutCreate : public CublasApiCAll {
  public:
    uint64_t virtual_ml;
    cudaDataType_t data_type;
    uint64_t rows;
    uint64_t cols;
    int64_t ld;

    CublasLtMatrixLayoutCreate(uint64_t virtualMl, cudaDataType_t dataType,
                               uint64_t rows, uint64_t cols, int64_t ld);
    explicit CublasLtMatrixLayoutCreate(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;
    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CublasLtMatrixLayoutDestroy : public CublasApiCAll {
  public:
    uint64_t virtual_ml;

    explicit CublasLtMatrixLayoutDestroy(uint64_t virtualMl);
    explicit CublasLtMatrixLayoutDestroy(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;
    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CublasLtMatrixLayoutSetAttribute : public CublasApiCAll {
  public:
    uint64_t virtual_ml;
    cublasLtMatrixLayoutAttribute_t attr;
    std::vector<uint8_t> buf;

    CublasLtMatrixLayoutSetAttribute(uint64_t virtualMl,
                                     cublasLtMatrixLayoutAttribute_t attr,
                                     std::vector<uint8_t> buf);
    explicit CublasLtMatrixLayoutSetAttribute(
        const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;
    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CublasSgemmStridedBatched : public CublasApiCAll {
  public:
    uint64_t virtual_handle;
    cublasOperation_t transa;
    cublasOperation_t transb;
    int m;
    int n;
    int k;
    float alpha;
    const float *A;
    int lda;
    long long int strideA;
    const float *B;
    int ldb;
    long long int strideB;
    float beta;
    float *C;
    int ldc;
    long long int strideC;
    int batchCount;

    CublasSgemmStridedBatched(uint64_t virtualHandle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              float alpha, const float *a, int lda,
                              long long int strideA, const float *b, int ldb,
                              long long int strideB, float beta, float *c,
                              int ldc, long long int strideC, int batchCount);

    explicit CublasSgemmStridedBatched(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;
    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CublasLtMatmulPreferenceCreate : public CublasApiCAll {
  public:
    uint64_t virtual_handle;

    explicit CublasLtMatmulPreferenceCreate(uint64_t virtualHandle);
    explicit CublasLtMatmulPreferenceCreate(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;
    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CublasLtMatmulPreferenceSetAttribute : public CublasApiCAll {
  public:
    uint64_t virtual_mmp;
    cublasLtMatmulPreferenceAttributes_t attr;
    std::vector<uint8_t> buf;

    CublasLtMatmulPreferenceSetAttribute(uint64_t virtualMmp,
                                         cublasLtMatmulPreferenceAttributes_t attr,
                                   std::vector<uint8_t> buf);
    explicit CublasLtMatmulPreferenceSetAttribute(
        const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;
    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CublasLtMatmulPreferenceDestroy : public CublasApiCAll {
  public:
    uint64_t virtual_mmp;

    explicit CublasLtMatmulPreferenceDestroy(uint64_t virtualMmp);
    explicit CublasLtMatmulPreferenceDestroy(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;
    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CublasLtMatmulAlgoGetHeuristic : public CublasApiCAll {
  public:
    uint64_t virtual_handle;
    uint64_t virtual_mmd;
    uint64_t virtual_ml_a_desc;
    uint64_t virtual_ml_b_desc;
    uint64_t virtual_ml_c_desc;
    uint64_t virtual_ml_d_desc;
    uint64_t virtual_mmp;
    uint64_t virtual_alg;

    CublasLtMatmulAlgoGetHeuristic(uint64_t virtualHandle, uint64_t virtualMmd,
                   uint64_t virtualMlADesc, uint64_t virtualMlBDesc,
                   uint64_t virtualMlCDesc, uint64_t virtualMlDDesc,
                   uint64_t virtual_mmp, uint64_t virtual_alg);
    explicit CublasLtMatmulAlgoGetHeuristic(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;
    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

} // namespace gpuless

#endif // GPULESS_CUBLAS_API_CALLS_HPP
