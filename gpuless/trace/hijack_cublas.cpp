#include "cublas_api_calls.hpp"
#include "libgpuless.hpp"

#include <cublas.h>
#include <cublasLt.h>

#define FAKE_HEURISTIC 503285028

namespace gpuless {

static uint64_t nextCublasHandle() {
    static uint64_t next = 1;
    return next++;
}

static uint64_t nextCublasLtHandle() {
    // TODO this is here because a cublas handle can also be used as a
    // cublasLtHandle
    static uint64_t next = 200000;
    return next++;
}

static uint64_t nextCublasLtMatmulDesc() {
    static uint64_t next = 1;
    return next++;
}

static uint64_t nextCublasLtMatrixLayout() {
    static uint64_t next = 1;
    return next++;
}

static uint64_t nextCublasLtMatmulPreferenceDesc() {
    static uint64_t next = 1;
    return next++;
}

static uint64_t nextCublasLtMatmulAlgo() {
    static uint64_t next = 1;
    return next++;
}

extern "C" {

cublasStatus_t cublasCreate_v2(cublasHandle_t *handle) {
    HIJACK_FN_PROLOGUE();
    auto virtual_handle = nextCublasHandle();
    *handle = reinterpret_cast<cublasHandle_t>(virtual_handle);
    getCudaTrace().record(std::make_shared<CublasCreateV2>(virtual_handle));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtCreate(cublasLtHandle_t *lighthandle) {
    HIJACK_FN_PROLOGUE();
    auto virtual_handle = nextCublasLtHandle();
    *lighthandle = reinterpret_cast<cublasLtHandle_t>(virtual_handle);
    getCudaTrace().record(std::make_shared<CublasLtCreate>(virtual_handle));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetStream_v2(cublasHandle_t handle,
                                  cudaStream_t streamId) {
    HIJACK_FN_PROLOGUE();
    getCudaTrace().record(std::make_shared<CublasSetStreamV2>(
        reinterpret_cast<uint64_t>(handle), streamId));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
    HIJACK_FN_PROLOGUE();
    getCudaTrace().record(std::make_shared<CublasSetMathMode>(
        reinterpret_cast<uint64_t>(handle), mode));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const float *alpha, const float *A, int lda,
                              const float *B, int ldb, const float *beta,
                              float *C, int ldc) {
    HIJACK_FN_PROLOGUE();
    getCudaTrace().record(std::make_shared<CublasSgemmV2>(
        reinterpret_cast<uint64_t>(handle), transa, transb, m, n, k, *alpha,
        *beta, A, B, C, lda, ldb, ldc));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc,
                                        cublasComputeType_t computeType,
                                        cudaDataType_t scaleType) {
    HIJACK_FN_PROLOGUE();
    auto virtual_handle = nextCublasLtMatmulDesc();
    *matmulDesc = reinterpret_cast<cublasLtMatmulDesc_t>(virtual_handle);
    getCudaTrace().record(std::make_shared<CublasLtMatmulDescCreate>(
        virtual_handle, computeType, scaleType));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc) {
    HIJACK_FN_PROLOGUE();
    getCudaTrace().record(std::make_shared<CublasLtMatmulDescDestroy>(
        reinterpret_cast<uint64_t>(matmulDesc)));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t
cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc,
                               cublasLtMatmulDescAttributes_t attr,
                               const void *buf, size_t sizeInBytes) {
    HIJACK_FN_PROLOGUE();
    std::vector<uint8_t> buf_vec(sizeInBytes);
    std::memcpy(buf_vec.data(), buf, sizeInBytes);
    getCudaTrace().record(std::make_shared<CublasLtMatmulDescSetAttribute>(
        reinterpret_cast<uint64_t>(matmulDesc), attr, buf_vec));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t
cublasLtMatmul(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc,
               const void *alpha, const void *A, cublasLtMatrixLayout_t Adesc,
               const void *B, cublasLtMatrixLayout_t Bdesc, const void *beta,
               const void *C, cublasLtMatrixLayout_t Cdesc, void *D,
               cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t *algo,
               void *workspace, size_t workspaceSizeInBytes,
               cudaStream_t stream) {
    HIJACK_FN_PROLOGUE();

    // TODO: determine actually correct size
    std::vector<uint8_t> alpha_vec(sizeof(float));
    std::vector<uint8_t> beta_vec(sizeof(float));
    std::memcpy(alpha_vec.data(), alpha, alpha_vec.size());
    std::memcpy(beta_vec.data(), beta, beta_vec.size());

    cublasLtMatmulAlgo_t algo_struct;
    if(algo) {
        algo_struct = *algo;
    }
    bool algo_is_null = (algo == nullptr);

    getCudaTrace().record(std::make_shared<CublasLtMatmul>(
        reinterpret_cast<uint64_t>(lightHandle),
        reinterpret_cast<uint64_t>(computeDesc), alpha_vec, beta_vec, A, B, C,
        D, reinterpret_cast<uint64_t>(Adesc), reinterpret_cast<uint64_t>(Bdesc),
        reinterpret_cast<uint64_t>(Cdesc), reinterpret_cast<uint64_t>(Ddesc),
        algo_struct, algo_is_null, workspace, workspaceSizeInBytes, stream));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout,
                                          cudaDataType type, uint64_t rows,
                                          uint64_t cols, int64_t ld) {
    HIJACK_FN_PROLOGUE();
    auto virtual_handle = nextCublasLtMatrixLayout();
    *matLayout = reinterpret_cast<cublasLtMatrixLayout_t>(virtual_handle);
    getCudaTrace().record(std::make_shared<CublasLtMatrixLayoutCreate>(
        virtual_handle, type, rows, cols, ld));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout) {
    HIJACK_FN_PROLOGUE();
    getCudaTrace().record(std::make_shared<CublasLtMatrixLayoutDestroy>(
        reinterpret_cast<uint64_t>(matLayout)));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t
cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout,
                                 cublasLtMatrixLayoutAttribute_t attr,
                                 const void *buf, size_t sizeInBytes) {
    HIJACK_FN_PROLOGUE();
    std::vector<uint8_t> attr_vec(sizeInBytes);
    std::memcpy(attr_vec.data(), buf, sizeInBytes);
    getCudaTrace().record(std::make_shared<CublasLtMatrixLayoutSetAttribute>(
        reinterpret_cast<uint64_t>(matLayout), attr, attr_vec));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t
cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                          cublasOperation_t transb, int m, int n, int k,
                          const float *alpha, const float *A, int lda,
                          long long int strideA, const float *B, int ldb,
                          long long int strideB, const float *beta, float *C,
                          int ldc, long long int strideC, int batchCount) {
    getCudaTrace().record(std::make_shared<CublasSgemmStridedBatched>(
        reinterpret_cast<uint64_t>(handle), transa, transb, m, n, k, *alpha, A,
        lda, strideA, B, ldb, strideB, *beta, C, ldc, strideC, batchCount));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc,
    cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc,
    cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc,
    cublasLtMatmulPreference_t preference, int requestedAlgoCount,
    cublasLtMatmulHeuristicResult_t heuristicResultsArray[],
    int *returnAlgoCount) {

    HIJACK_FN_PROLOGUE();
    auto virtual_alg = nextCublasLtMatmulAlgo();

    cublasLtMatmulHeuristicResult_t virtual_res{
        {virtual_alg, 0, 0, 0, 0, 0, 0, 24242},
        0,
        CUBLAS_STATUS_SUCCESS,
        0.2f,
        {0, 0, 0, 0}};

    heuristicResultsArray[0] = virtual_res;
    *returnAlgoCount = 1;

    getCudaTrace().record(std::make_shared<CublasLtMatmulAlgoGetHeuristic>(
        reinterpret_cast<uint64_t>(lightHandle),
        reinterpret_cast<uint64_t>(operationDesc),
        reinterpret_cast<uint64_t>(Adesc), reinterpret_cast<uint64_t>(Bdesc),
        reinterpret_cast<uint64_t>(Cdesc), reinterpret_cast<uint64_t>(Ddesc),
        reinterpret_cast<uint64_t>(preference), virtual_alg));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t
cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref) {
    HIJACK_FN_PROLOGUE();
    auto virtual_handle = nextCublasLtMatmulPreferenceDesc();
    *pref = reinterpret_cast<cublasLtMatmulPreference_t>(virtual_handle);
    getCudaTrace().record(
        std::make_shared<CublasLtMatmulPreferenceCreate>(virtual_handle));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t
cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref,
                                     cublasLtMatmulPreferenceAttributes_t attr,
                                     const void *buf, size_t sizeInBytes) {
    HIJACK_FN_PROLOGUE();
    std::vector<uint8_t> attr_vec(sizeInBytes);
    std::memcpy(attr_vec.data(), buf, sizeInBytes);
    getCudaTrace().record(
        std::make_shared<CublasLtMatmulPreferenceSetAttribute>(
            reinterpret_cast<uint64_t>(pref), attr, attr_vec));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t
cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref) {
    HIJACK_FN_PROLOGUE();
    getCudaTrace().record(std::make_shared<CublasLtMatmulPreferenceDestroy>(
        reinterpret_cast<uint64_t>(pref)));
    return CUBLAS_STATUS_SUCCESS;
}
}

} // namespace gpuless
