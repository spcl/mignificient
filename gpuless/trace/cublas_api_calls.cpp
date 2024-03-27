#include "cublas_api_calls.hpp"

#include "../schemas/cublas_calls_generated.h"
#include "../schemas/trace_execution_protocol_generated.h"
#include "libgpuless.hpp"
#include <utility>

namespace gpuless {

std::string gpuless::CublasApiCAll::nativeErrorToString(uint64_t err) {
    auto status = static_cast<cublasStatus_t>(err);
    std::string str_err;

    switch (status) {
    case CUBLAS_STATUS_SUCCESS:
        str_err = "CUBLAS_STATUS_SUCCESS";
        break;
    case CUBLAS_STATUS_NOT_INITIALIZED:
        str_err = "CUBLAS_STATUS_NOT_INITIALIZED";
        break;
    case CUBLAS_STATUS_ALLOC_FAILED:
        str_err = "CUBLAS_STATUS_ALLOC_FAILED";
        break;
    case CUBLAS_STATUS_INVALID_VALUE:
        str_err = "CUBLAS_STATUS_INVALID_VALUE";
        break;
    case CUBLAS_STATUS_ARCH_MISMATCH:
        str_err = "CUBLAS_STATUS_ARCH_MISMATCH";
        break;
    case CUBLAS_STATUS_MAPPING_ERROR:
        str_err = "CUBLAS_STATUS_MAPPING_ERROR";
        break;
    case CUBLAS_STATUS_EXECUTION_FAILED:
        str_err = "CUBLAS_STATUS_EXECUTION_FAILED";
        break;
    case CUBLAS_STATUS_INTERNAL_ERROR:
        str_err = "CUBLAS_STATUS_INTERNAL_ERROR";
        break;
    case CUBLAS_STATUS_NOT_SUPPORTED:
        str_err = "CUBLAS_STATUS_NOT_SUPPORTED";
        break;
    case CUBLAS_STATUS_LICENSE_ERROR:
        str_err = "CUBLAS_STATUS_LICENSE_ERROR";
        break;
    }

    return "[cublas] " + str_err;
}

/*
 * cublasCreate_v2
 */
gpuless::CublasCreateV2::CublasCreateV2(uint64_t virtualHandle)
    : virtual_handle(virtualHandle) {}

uint64_t gpuless::CublasCreateV2::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cublasCreate_v2);
    if (vdev.cublas_handle_virtual_to_real.size() < this->virtual_handle + 1) {
        vdev.cublas_handle_virtual_to_real.resize(this->virtual_handle + 1);
    }
    if (vdev.cublaslt_handle_virtual_to_real.size() <
        this->virtual_handle + 1) {
        vdev.cublaslt_handle_virtual_to_real.resize(this->virtual_handle + 1);
    }
    cublasStatus_t res =
        real(&vdev.cublas_handle_virtual_to_real[this->virtual_handle]);
    vdev.cublaslt_handle_virtual_to_real[this->virtual_handle] =
        reinterpret_cast<cublasLtHandle_t>(
            vdev.cublas_handle_virtual_to_real[this->virtual_handle]);
    return res;
}

flatbuffers::Offset<FBCudaApiCall>
gpuless::CublasCreateV2::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCublasCreateV2(builder, this->virtual_handle);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCublasCreateV2, api_call.Union());
    return api_call_union;
}

CublasCreateV2::CublasCreateV2(const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCublasCreateV2();
    this->virtual_handle = c->virtual_handle();
}

/*
 * cublasSetStream_v2
 */
gpuless::CublasSetStreamV2::CublasSetStreamV2(uint64_t virtualHandle,
                                              cudaStream_t stream)
    : virtual_handle(virtualHandle), stream(stream) {}

uint64_t gpuless::CublasSetStreamV2::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cublasSetStream_v2);
    return real(vdev.cublas_handle_virtual_to_real[this->virtual_handle],
                this->stream);
}

flatbuffers::Offset<FBCudaApiCall>
CublasSetStreamV2::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {
    auto api_call =
        CreateFBCublasSetStreamV2(builder, this->virtual_handle,
                                  reinterpret_cast<uint64_t>(this->stream));
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCublasSetStreamV2, api_call.Union());
    return api_call_union;
}

CublasSetStreamV2::CublasSetStreamV2(const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCublasSetStreamV2();
    this->virtual_handle = c->virtual_handle();
    this->stream = reinterpret_cast<cudaStream_t>(c->stream());
}

/*
 * cublasSetMathMode
 */
gpuless::CublasSetMathMode::CublasSetMathMode(uint64_t virtualHandle,
                                              cublasMath_t mode)
    : virtual_handle(virtualHandle), mode(mode) {}

uint64_t gpuless::CublasSetMathMode::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cublasSetMathMode);
    return real(vdev.cublas_handle_virtual_to_real[this->virtual_handle],
                this->mode);
}

flatbuffers::Offset<FBCudaApiCall>
CublasSetMathMode::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {
    auto api_call =
        CreateFBCublasSetMathMode(builder, this->virtual_handle, this->mode);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCublasSetMathMode, api_call.Union());
    return api_call_union;
}

CublasSetMathMode::CublasSetMathMode(const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCublasSetMathMode();
    this->virtual_handle = c->virtual_handle();
    this->mode = static_cast<cublasMath_t>(c->math_mode());
}

/*
 * cublasSgemm_v2
 */
gpuless::CublasSgemmV2::CublasSgemmV2(uint64_t virtualHandle,
                                      cublasOperation_t transa,
                                      cublasOperation_t transb, int m, int n,
                                      int k, float alpha, float beta,
                                      const float *a, const float *b,
                                      const float *c, int lda, int ldb, int ldc)
    : virtual_handle(virtualHandle), transa(transa), transb(transb), m(m), n(n),
      k(k), alpha(alpha), beta(beta), A(a), B(b), C(c), lda(lda), ldb(ldb),
      ldc(ldc) {}

uint64_t gpuless::CublasSgemmV2::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cublasSgemm_v2);
    return real(vdev.cublas_handle_virtual_to_real[this->virtual_handle],
                this->transa, this->transb, this->m, this->n, this->k,
                &this->alpha, this->A, this->lda, this->B, this->ldb,
                &this->beta, const_cast<float *>(this->C), this->ldc);
}

flatbuffers::Offset<FBCudaApiCall>
CublasSgemmV2::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCublasSgemmV2(
        builder, this->virtual_handle, this->transa, this->transb, this->m,
        this->n, this->k, this->alpha, this->beta,
        reinterpret_cast<uint64_t>(this->A),
        reinterpret_cast<uint64_t>(this->B),
        reinterpret_cast<uint64_t>(this->C), this->lda, this->ldb, this->ldc);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCublasSgemmV2, api_call.Union());
    return api_call_union;
}

CublasSgemmV2::CublasSgemmV2(const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCublasSgemmV2();
    this->virtual_handle = c->virtual_handle();
    this->transa = static_cast<cublasOperation_t>(c->transa_op());
    this->transb = static_cast<cublasOperation_t>(c->transb_op());
    this->m = c->m();
    this->n = c->n();
    this->k = c->k();
    this->alpha = c->alpha();
    this->beta = c->beta();
    this->A = reinterpret_cast<const float *>(c->a());
    this->B = reinterpret_cast<const float *>(c->b());
    this->C = reinterpret_cast<const float *>(c->c());
    this->lda = c->lda();
    this->ldb = c->ldb();
    this->ldc = c->ldc();
}

/*
 * cublasLtCreate
 */
CublasLtCreate::CublasLtCreate(uint64_t virtualHandle)
    : virtual_handle(virtualHandle) {}

uint64_t CublasLtCreate::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cublasLtCreate);
    if (vdev.cublaslt_handle_virtual_to_real.size() <
        this->virtual_handle + 1) {
        vdev.cublaslt_handle_virtual_to_real.resize(this->virtual_handle + 1);
    }
    return real(&vdev.cublaslt_handle_virtual_to_real[this->virtual_handle]);
}

CublasLtCreate::CublasLtCreate(const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCublasLtCreate();
    this->virtual_handle = c->virtual_handle();
}

flatbuffers::Offset<FBCudaApiCall>
CublasLtCreate::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCublasLtCreate(builder, this->virtual_handle);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCublasLtCreate, api_call.Union());
    return api_call_union;
}

/*
 * cublasLtMatmulDescCreate
 */
CublasLtMatmulDescCreate::CublasLtMatmulDescCreate(
    uint64_t virtualMmd, cublasComputeType_t computeType,
    cudaDataType_t scaleType)
    : virtual_mmd(virtualMmd), compute_type(computeType),
      scale_type(scaleType) {}

CublasLtMatmulDescCreate::CublasLtMatmulDescCreate(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCublasLtMatmulDescCreate();
    this->virtual_mmd = c->virtual_mmd();
    this->compute_type = static_cast<cublasComputeType_t>(c->compute_type());
    this->scale_type = static_cast<cudaDataType_t>(c->scale_type());
}

uint64_t CublasLtMatmulDescCreate::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cublasLtMatmulDescCreate);
    if (vdev.cublaslt_matmul_handle_virtual_to_real.size() <
        this->virtual_mmd + 1) {
        vdev.cublaslt_matmul_handle_virtual_to_real.resize(this->virtual_mmd +
                                                           1);
    }
    return real(&vdev.cublaslt_matmul_handle_virtual_to_real[this->virtual_mmd],
                this->compute_type, this->scale_type);
}

flatbuffers::Offset<FBCudaApiCall>
CublasLtMatmulDescCreate::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCublasLtMatmulDescCreate(
        builder, this->virtual_mmd, this->compute_type, this->scale_type);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCublasLtMatmulDescCreate,
        api_call.Union());
    return api_call_union;
}

/*
 * cublasLtMatmulDescDestroy
 */
CublasLtMatmulDescDestroy::CublasLtMatmulDescDestroy(uint64_t virtualMmd)
    : virtual_mmd(virtualMmd) {}

CublasLtMatmulDescDestroy::CublasLtMatmulDescDestroy(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCublasLtMatmulDescDestroy();
    this->virtual_mmd = c->virtual_mmd();
}

uint64_t CublasLtMatmulDescDestroy::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cublasLtMatmulDescDestroy);
    return real(vdev.cublaslt_matmul_handle_virtual_to_real[this->virtual_mmd]);
}

flatbuffers::Offset<FBCudaApiCall> CublasLtMatmulDescDestroy::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    auto api_call =
        CreateFBCublasLtMatmulDescDestroy(builder, this->virtual_mmd);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCublasLtMatmulDescDestroy,
        api_call.Union());
    return api_call_union;
}

/*
 * cublasLtMatmulDescSetAttribute
 */

CublasLtMatmulDescSetAttribute::CublasLtMatmulDescSetAttribute(
    uint64_t virtualMmd, cublasLtMatmulDescAttributes_t attr,
    std::vector<uint8_t> buf)
    : virtual_mmd(virtualMmd), attr(attr), buf(std::move(buf)) {}

CublasLtMatmulDescSetAttribute::CublasLtMatmulDescSetAttribute(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCublasLtMatmulDescSetAttribute();
    this->virtual_mmd = c->virtual_mmd();
    this->attr = static_cast<cublasLtMatmulDescAttributes_t>(c->attr());
    this->buf.resize(c->buf()->size());
    std::memcpy(this->buf.data(), c->buf()->data(), c->buf()->size());
}

uint64_t
CublasLtMatmulDescSetAttribute::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cublasLtMatmulDescSetAttribute);
    return real(vdev.cublaslt_matmul_handle_virtual_to_real[this->virtual_mmd],
                this->attr, this->buf.data(), this->buf.size());
}

flatbuffers::Offset<FBCudaApiCall> CublasLtMatmulDescSetAttribute::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCublasLtMatmulDescSetAttribute(
        builder, this->virtual_mmd, this->attr,
        builder.CreateVector(this->buf));
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCublasLtMatmulDescSetAttribute,
        api_call.Union());
    return api_call_union;
}

/*
 * cublasLtMatmul
 */
CublasLtMatmul::CublasLtMatmul(uint64_t virtualHandle, uint64_t virtualMmd,
                               std::vector<uint8_t> &alpha,
                               std::vector<uint8_t> &beta, const void *a,
                               const void *b, const void *c, void *d,
                               uint64_t virtualMlADesc, uint64_t virtualMlBDesc,
                               uint64_t virtualMlCDesc, uint64_t virtualMlDDesc,
                               const cublasLtMatmulAlgo_t &algo,
                               bool algoIsNull, void *workspace,
                               size_t workspaceSizeInBytes, cudaStream_t stream)
    : virtual_handle(virtualHandle), virtual_mmd(virtualMmd), alpha(alpha),
      beta(beta), A(a), B(b), C(c), D(d), virtual_ml_a_desc(virtualMlADesc),
      virtual_ml_b_desc(virtualMlBDesc), virtual_ml_c_desc(virtualMlCDesc),
      virtual_ml_d_desc(virtualMlDDesc), algo(algo), algo_is_null(algoIsNull),
      workspace(workspace), workspace_size_in_bytes(workspaceSizeInBytes),
      stream(stream) {}

CublasLtMatmul::CublasLtMatmul(const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCublasLtMatmul();
    this->virtual_handle = c->virtual_handle();
    this->virtual_mmd = c->virtual_mmd();
    this->alpha.resize(c->alpha()->size());
    std::memcpy(this->alpha.data(), c->alpha()->data(), c->alpha()->size());
    this->beta.resize(c->beta()->size());
    std::memcpy(this->beta.data(), c->beta()->data(), c->beta()->size());
    this->A = reinterpret_cast<const void *>(c->a());
    this->B = reinterpret_cast<const void *>(c->b());
    this->C = reinterpret_cast<const void *>(c->c());
    this->D = reinterpret_cast<void *>(c->d());
    this->virtual_ml_a_desc = c->virtual_ml_a_desc();
    this->virtual_ml_b_desc = c->virtual_ml_b_desc();
    this->virtual_ml_c_desc = c->virtual_ml_c_desc();
    this->virtual_ml_d_desc = c->virtual_ml_d_desc();

    assert(c->algo()->size() >= 8);
    this->algo.data[0] = c->algo()->Get(0);
    this->algo.data[1] = c->algo()->Get(1);
    this->algo.data[2] = c->algo()->Get(2);
    this->algo.data[3] = c->algo()->Get(3);
    this->algo.data[4] = c->algo()->Get(4);
    this->algo.data[5] = c->algo()->Get(5);
    this->algo.data[6] = c->algo()->Get(6);
    this->algo.data[7] = c->algo()->Get(7);

    this->algo_is_null = c->algo_is_null() != 0;
    this->workspace = reinterpret_cast<void *>(c->workspace());
    this->workspace_size_in_bytes = c->workspace_size_in_bytes();
    this->stream = reinterpret_cast<cudaStream_t>(c->stream());
}

uint64_t CublasLtMatmul::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cublasLtMatmul);

    cublasLtMatmulAlgo_t *algo_ptr = nullptr;
    if (!this->algo_is_null) {
        algo_ptr = &this->algo;
    }

    void *ws = this->workspace;
    std::size_t ws_size = this->workspace_size_in_bytes;

    if (!this->algo_is_null && this->algo.data[7] == 24242) {
        algo_ptr = &(
            vdev.cublaslt_matmul_alg_virtual_to_real[this->algo.data[0]].algo);
        std::size_t algo_sz =
            vdev.cublaslt_matmul_alg_virtual_to_real[this->algo.data[0]]
                .workspaceSize;
        if (algo_sz > this->workspace_size_in_bytes) {
            ws_size =
                vdev.cublaslt_matmul_alg_virtual_to_real[this->algo.data[0]]
                    .workspaceSize;
            ws = vdev.get_scratch(ws_size);
        } else {
            ws = this->workspace;
        }
    }

    auto err =
        real(vdev.cublaslt_handle_virtual_to_real[this->virtual_handle],
             vdev.cublaslt_matmul_handle_virtual_to_real[this->virtual_mmd],
             this->alpha.data(), this->A,
             vdev.cublaslt_matrix_layout_handle_virtual_to_real
                 [this->virtual_ml_a_desc],
             this->B,
             vdev.cublaslt_matrix_layout_handle_virtual_to_real
                 [this->virtual_ml_b_desc],
             this->beta.data(), this->C,
             vdev.cublaslt_matrix_layout_handle_virtual_to_real
                 [this->virtual_ml_c_desc],
             this->D,
             vdev.cublaslt_matrix_layout_handle_virtual_to_real
                 [this->virtual_ml_d_desc],
             algo_ptr, ws, ws_size, this->stream);

    vdev.free_scratch();
    return err;
}

flatbuffers::Offset<FBCudaApiCall>
CublasLtMatmul::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {
    std::vector<uint64_t> algo_vec(8);
    std::memcpy(algo_vec.data(), this->algo.data, 8 * sizeof(uint64_t));
    auto api_call = CreateFBCublasLtMatmul(
        builder, this->virtual_handle, this->virtual_mmd,
        builder.CreateVector(this->alpha), builder.CreateVector(this->beta),
        reinterpret_cast<uint64_t>(this->A),
        reinterpret_cast<uint64_t>(this->B),
        reinterpret_cast<uint64_t>(this->C),
        reinterpret_cast<uint64_t>(this->D), this->virtual_ml_a_desc,
        this->virtual_ml_b_desc, this->virtual_ml_c_desc,
        this->virtual_ml_d_desc, builder.CreateVector(algo_vec),
        static_cast<uint64_t>(this->algo_is_null),
        reinterpret_cast<uint64_t>(this->workspace),
        this->workspace_size_in_bytes,
        reinterpret_cast<uint64_t>(this->stream));
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCublasLtMatmul, api_call.Union());
    return api_call_union;
}

/*
 * cublasLtMatrixLayoutCreate
 */
CublasLtMatrixLayoutCreate::CublasLtMatrixLayoutCreate(uint64_t virtualMl,
                                                       cudaDataType_t dataType,
                                                       uint64_t rows,
                                                       uint64_t cols,
                                                       int64_t ld)
    : virtual_ml(virtualMl), data_type(dataType), rows(rows), cols(cols),
      ld(ld) {}

CublasLtMatrixLayoutCreate::CublasLtMatrixLayoutCreate(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCublasLtMatrixLayoutCreate();
    this->virtual_ml = c->virtual_ml();
    this->data_type = static_cast<cudaDataType_t>(c->data_type());
    this->rows = c->rows();
    this->cols = c->cols();
    this->ld = c->ld();
}

uint64_t CublasLtMatrixLayoutCreate::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cublasLtMatrixLayoutCreate);
    if (vdev.cublaslt_matrix_layout_handle_virtual_to_real.size() <
        this->virtual_ml + 1) {
        vdev.cublaslt_matrix_layout_handle_virtual_to_real.resize(
            this->virtual_ml + 1);
    }
    return real(
        &vdev.cublaslt_matrix_layout_handle_virtual_to_real[this->virtual_ml],
        this->data_type, this->rows, this->cols, this->ld);
}

flatbuffers::Offset<FBCudaApiCall> CublasLtMatrixLayoutCreate::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCublasLtMatrixLayoutCreate(
        builder, this->virtual_ml, this->data_type, this->rows, this->cols,
        this->ld);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCublasLtMatrixLayoutCreate,
        api_call.Union());
    return api_call_union;
}

/*
 * cublasLtMatrixLayoutDestroy
 */
CublasLtMatrixLayoutDestroy::CublasLtMatrixLayoutDestroy(uint64_t virtualMl)
    : virtual_ml(virtualMl) {}

CublasLtMatrixLayoutDestroy::CublasLtMatrixLayoutDestroy(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCublasLtMatrixLayoutDestroy();
    this->virtual_ml = c->virtual_ml();
}

uint64_t CublasLtMatrixLayoutDestroy::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cublasLtMatrixLayoutDestroy);
    return real(
        vdev.cublaslt_matrix_layout_handle_virtual_to_real[this->virtual_ml]);
}

flatbuffers::Offset<FBCudaApiCall> CublasLtMatrixLayoutDestroy::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    auto api_call =
        CreateFBCublasLtMatrixLayoutDestroy(builder, this->virtual_ml);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCublasLtMatrixLayoutDestroy,
        api_call.Union());
    return api_call_union;
}

/*
 * CublasLtMatrixLayoutSetAttribute
 */
CublasLtMatrixLayoutSetAttribute::CublasLtMatrixLayoutSetAttribute(
    uint64_t virtualMl, cublasLtMatrixLayoutAttribute_t attr,
    std::vector<uint8_t> buf)
    : virtual_ml(virtualMl), attr(attr), buf(std::move(buf)) {}

CublasLtMatrixLayoutSetAttribute::CublasLtMatrixLayoutSetAttribute(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCublasLtMatrixLayoutSetAttribute();
    this->virtual_ml = c->virtual_ml();
    this->attr = static_cast<cublasLtMatrixLayoutAttribute_t>(c->attr());
    this->buf.resize(c->buf()->size());
    std::memcpy(this->buf.data(), c->buf()->data(), c->buf()->size());
}

uint64_t
CublasLtMatrixLayoutSetAttribute::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cublasLtMatrixLayoutSetAttribute);
    return real(
        vdev.cublaslt_matrix_layout_handle_virtual_to_real[this->virtual_ml],
        this->attr, this->buf.data(), this->buf.size());
}

flatbuffers::Offset<FBCudaApiCall>
CublasLtMatrixLayoutSetAttribute::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCublasLtMatrixLayoutSetAttribute(
        builder, this->virtual_ml, this->attr, builder.CreateVector(this->buf));
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCublasLtMatrixLayoutSetAttribute,
        api_call.Union());
    return api_call_union;
}

/*
 * CublasSgemmStridedBatched
 */
CublasSgemmStridedBatched::CublasSgemmStridedBatched(
    uint64_t virtualHandle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float alpha, const float *a, int lda,
    long long int strideA, const float *b, int ldb, long long int strideB,
    const float beta, float *c, int ldc, long long int strideC, int batchCount)
    : virtual_handle(virtualHandle), transa(transa), transb(transb), m(m), n(n),
      k(k), alpha(alpha), A(a), lda(lda), strideA(strideA), B(b), ldb(ldb),
      strideB(strideB), beta(beta), C(c), ldc(ldc), strideC(strideC),
      batchCount(batchCount) {}

CublasSgemmStridedBatched::CublasSgemmStridedBatched(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCublasSgemmStridedBatched();
    this->virtual_handle = c->virtual_handle();
    this->transa = static_cast<cublasOperation_t>(c->transa());
    this->transb = static_cast<cublasOperation_t>(c->transb());
    this->m = c->m();
    this->n = c->n();
    this->k = c->k();
    this->alpha = c->alpha();
    this->A = reinterpret_cast<const float *>(c->a());
    this->lda = c->lda();
    this->strideA = c->stride_a();
    this->B = reinterpret_cast<const float *>(c->b());
    this->ldb = c->ldb();
    this->strideB = c->stride_b();
    this->beta = c->beta();
    this->C = reinterpret_cast<float *>(c->c());
    this->ldc = c->ldc();
    this->strideC = c->stride_c();
    this->batchCount = c->batch_count();
}

uint64_t CublasSgemmStridedBatched::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cublasSgemmStridedBatched);
    cublasHandle_t handle =
        vdev.cublas_handle_virtual_to_real[this->virtual_handle];
    return real(handle, this->transa, this->transb, this->m, this->n, this->k,
                &this->alpha, this->A, this->lda, this->strideA, this->B,
                this->ldb, this->strideB, &this->beta, this->C, this->ldc,
                this->strideC, this->batchCount);
}

flatbuffers::Offset<FBCudaApiCall> CublasSgemmStridedBatched::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCublasSgemmStridedBatched(
        builder, this->virtual_handle, this->transa, this->transb, this->m,
        this->n, this->k, this->alpha, reinterpret_cast<uint64_t>(this->A),
        this->lda, this->strideA, reinterpret_cast<uint64_t>(this->B),
        this->ldb, this->strideB, this->beta,
        reinterpret_cast<uint64_t>(this->C), this->ldc, this->strideC,
        this->batchCount);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCublasSgemmStridedBatched,
        api_call.Union());
    return api_call_union;
}
/*
 * cublasLtMatmulPreferenceCreate
 */
CublasLtMatmulPreferenceCreate::CublasLtMatmulPreferenceCreate(
    uint64_t virtualHandle)
    : virtual_handle(virtualHandle) {}

CublasLtMatmulPreferenceCreate::CublasLtMatmulPreferenceCreate(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCublasLtMatmulPreferenceCreate();
    this->virtual_handle = c->virtual_mmp();
}

uint64_t
CublasLtMatmulPreferenceCreate::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cublasLtMatmulPreferenceCreate);
    if (vdev.cublaslt_matmul_pref_handle_virtual_to_real.size() <
        this->virtual_handle + 1) {
        vdev.cublaslt_matmul_pref_handle_virtual_to_real.resize(
            this->virtual_handle + 1);
    }
    return real(&vdev.cublaslt_matmul_pref_handle_virtual_to_real
                     [this->virtual_handle]);
}

flatbuffers::Offset<FBCudaApiCall> CublasLtMatmulPreferenceCreate::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    auto api_call =
        CreateFBCublasLtMatmulPreferenceCreate(builder, this->virtual_handle);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCublasLtMatmulPreferenceCreate,
        api_call.Union());
    return api_call_union;
}

/*
 * cublasLtMatmulPreferenceSetAttribute
 */

CublasLtMatmulPreferenceSetAttribute::CublasLtMatmulPreferenceSetAttribute(
    uint64_t virtualMmp, cublasLtMatmulPreferenceAttributes_t attr,
    std::vector<uint8_t> buf)
    : virtual_mmp(virtualMmp), attr(attr), buf(std::move(buf)) {}

CublasLtMatmulPreferenceSetAttribute::CublasLtMatmulPreferenceSetAttribute(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c =
        fb_cuda_api_call->api_call_as_FBCublasLtMatmulPreferenceSetAttribute();
    this->virtual_mmp = c->virtual_mmp();
    this->attr = static_cast<cublasLtMatmulPreferenceAttributes_t>(c->attr());
    this->buf.resize(c->buf()->size());
    std::memcpy(this->buf.data(), c->buf()->data(), c->buf()->size());
}

uint64_t
CublasLtMatmulPreferenceSetAttribute::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cublasLtMatmulPreferenceSetAttribute);
    return real(
        vdev.cublaslt_matmul_pref_handle_virtual_to_real[this->virtual_mmp],
        this->attr, this->buf.data(), this->buf.size());
}

flatbuffers::Offset<FBCudaApiCall>
CublasLtMatmulPreferenceSetAttribute::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCublasLtMatmulPreferenceSetAttribute(
        builder, this->virtual_mmp, this->attr,
        builder.CreateVector(this->buf));
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCublasLtMatmulPreferenceSetAttribute,
        api_call.Union());
    return api_call_union;
}

/*
 * cublasLtMatmulPreferenceDestroy
 */
CublasLtMatmulPreferenceDestroy::CublasLtMatmulPreferenceDestroy(
    uint64_t virtualMmp)
    : virtual_mmp(virtualMmp) {}

CublasLtMatmulPreferenceDestroy::CublasLtMatmulPreferenceDestroy(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCublasLtMatmulPreferenceDestroy();
    this->virtual_mmp = c->virtual_mmp();
}

uint64_t
CublasLtMatmulPreferenceDestroy::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cublasLtMatmulPreferenceDestroy);
    return real(
        vdev.cublaslt_matmul_pref_handle_virtual_to_real[this->virtual_mmp]);
}

flatbuffers::Offset<FBCudaApiCall> CublasLtMatmulPreferenceDestroy::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    auto api_call =
        CreateFBCublasLtMatmulPreferenceDestroy(builder, this->virtual_mmp);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCublasLtMatmulPreferenceDestroy,
        api_call.Union());
    return api_call_union;
}

/*
 * cublasLtMatmulAlgoGetHeuristic
 */
CublasLtMatmulAlgoGetHeuristic::CublasLtMatmulAlgoGetHeuristic(
    uint64_t virtualHandle, uint64_t virtualMmd, uint64_t virtualMlADesc,
    uint64_t virtualMlBDesc, uint64_t virtualMlCDesc, uint64_t virtualMlDDesc,
    uint64_t virtualMmp, uint64_t virtualAlg)
    : virtual_handle(virtualHandle), virtual_mmd(virtualMmd),
      virtual_ml_a_desc(virtualMlADesc), virtual_ml_b_desc(virtualMlBDesc),
      virtual_ml_c_desc(virtualMlCDesc), virtual_ml_d_desc(virtualMlDDesc),
      virtual_mmp(virtualMmp), virtual_alg(virtualAlg) {}

CublasLtMatmulAlgoGetHeuristic::CublasLtMatmulAlgoGetHeuristic(
    const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCublasLtMatmulAlgoGetHeuristic();
    this->virtual_handle = c->virtual_handle();
    this->virtual_mmd = c->virtual_mmd();
    this->virtual_ml_a_desc = c->virtual_ml_a_desc();
    this->virtual_ml_b_desc = c->virtual_ml_b_desc();
    this->virtual_ml_c_desc = c->virtual_ml_c_desc();
    this->virtual_ml_d_desc = c->virtual_ml_d_desc();
    this->virtual_mmp = c->virtual_mmp();
    this->virtual_alg = c->virtual_alg();
}

uint64_t
CublasLtMatmulAlgoGetHeuristic::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cublasLtMatmulAlgoGetHeuristic);

    std::vector<cublasLtMatmulHeuristicResult_t> algs(6);
    int returnedAlgs;

    auto err = real(
        vdev.cublaslt_handle_virtual_to_real[this->virtual_handle],
        vdev.cublaslt_matmul_handle_virtual_to_real[this->virtual_mmd],
        vdev.cublaslt_matrix_layout_handle_virtual_to_real
            [this->virtual_ml_a_desc],
        vdev.cublaslt_matrix_layout_handle_virtual_to_real
            [this->virtual_ml_b_desc],
        vdev.cublaslt_matrix_layout_handle_virtual_to_real
            [this->virtual_ml_c_desc],
        vdev.cublaslt_matrix_layout_handle_virtual_to_real
            [this->virtual_ml_d_desc],
        vdev.cublaslt_matmul_pref_handle_virtual_to_real[this->virtual_mmp], 6,
        algs.data(), &returnedAlgs);

    if (returnedAlgs < 1)
        throw std::runtime_error("Matmul returned less than one algorithm");

    if (vdev.cublaslt_matmul_alg_virtual_to_real.size() <
        this->virtual_alg + 1) {
        vdev.cublaslt_matmul_alg_virtual_to_real.resize(this->virtual_alg + 1);
    }

    vdev.cublaslt_matmul_alg_virtual_to_real[this->virtual_alg] = algs[0];

    return err;
}

flatbuffers::Offset<FBCudaApiCall> CublasLtMatmulAlgoGetHeuristic::fbSerialize(
    flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCublasLtMatmulAlgoGetHeuristic(
        builder, this->virtual_handle, this->virtual_mmd,
        this->virtual_ml_a_desc, this->virtual_ml_b_desc,
        this->virtual_ml_c_desc, this->virtual_ml_d_desc, this->virtual_mmp,
        this->virtual_alg);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCublasLtMatmulAlgoGetHeuristic,
        api_call.Union());
    return api_call_union;
}

} // namespace gpuless
