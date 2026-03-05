// cublas_gemm.cu — cuBLAS/cublasLt GEMM backend implementing gemm.h
//
// Uses cublasLt with algorithm caching for optimal performance.
// Links against -lcublas.

#include "gemm.h"
#include "kernels.h"

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e) { fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)
#define CUBLAS_CHECK(call) do { cublasStatus_t stat = (call); if (stat != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, (int)stat); exit(1); } } while(0)

// =========================================================================
// Static state
// =========================================================================

static cublasHandle_t   s_cublas   = nullptr;
static cublasLtHandle_t s_cublaslt = nullptr;
static void*  s_workspace      = nullptr;
static size_t s_workspace_size = 0;
static cudaStream_t s_stream   = nullptr;

// =========================================================================
// cublasLt algorithm cache
// =========================================================================

struct GemmKey {
    int m, n, k;
    cublasOperation_t opA, opB;
    bool accumulate;
    bool has_bias;
    bool operator==(const GemmKey& o) const {
        return m == o.m && n == o.n && k == o.k && opA == o.opA && opB == o.opB
            && accumulate == o.accumulate && has_bias == o.has_bias;
    }
};
struct GemmKeyHash {
    size_t operator()(const GemmKey& k) const {
        size_t h = std::hash<int>()(k.m);
        h ^= std::hash<int>()(k.n) * 2654435761ULL;
        h ^= std::hash<int>()(k.k) * 40343;
        h ^= std::hash<int>()(k.opA) * 73;
        h ^= std::hash<int>()(k.opB) * 131;
        h ^= std::hash<int>()(k.accumulate) * 257;
        h ^= std::hash<int>()(k.has_bias) * 521;
        return h;
    }
};
struct GemmPlan {
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t layout_A, layout_B, layout_C;
    cublasLtMatmulAlgo_t algo;
    bool valid;
};
static std::unordered_map<GemmKey, GemmPlan, GemmKeyHash> s_gemm_cache;

// =========================================================================
// Lifecycle
// =========================================================================

void gemm_init(cudaStream_t stream) {
    s_stream = stream;
    CUBLAS_CHECK(cublasCreate(&s_cublas));
    CUBLAS_CHECK(cublasSetStream(s_cublas, stream));
    CUBLAS_CHECK(cublasSetMathMode(s_cublas, CUBLAS_TENSOR_OP_MATH));
    CUBLAS_CHECK(cublasLtCreate(&s_cublaslt));

    s_workspace_size = 32 * 1024 * 1024;  // 32MB
    CUDA_CHECK(cudaMalloc(&s_workspace, s_workspace_size));
    CUBLAS_CHECK(cublasSetWorkspace(s_cublas, s_workspace, s_workspace_size));
}

void gemm_free() {
    if (s_cublas)   { cublasDestroy(s_cublas);   s_cublas = nullptr; }
    if (s_cublaslt) { cublasLtDestroy(s_cublaslt); s_cublaslt = nullptr; }
    for (auto& [key, plan] : s_gemm_cache) {
        cublasLtMatmulDescDestroy(plan.matmul_desc);
        cublasLtMatrixLayoutDestroy(plan.layout_A);
        cublasLtMatrixLayoutDestroy(plan.layout_B);
        cublasLtMatrixLayoutDestroy(plan.layout_C);
    }
    s_gemm_cache.clear();
    if (s_workspace) { cudaFree(s_workspace); s_workspace = nullptr; }
}

// =========================================================================
// cublasLt GEMM with algorithm caching
// =========================================================================

static GemmPlan& get_gemm_plan(int col_m, int col_n, int col_k,
                                int ldA, int ldB, int ldC,
                                cublasOperation_t opA, cublasOperation_t opB,
                                bool accumulate, bool has_bias) {
    GemmKey key{col_m, col_n, col_k, opA, opB, accumulate, has_bias};
    auto it = s_gemm_cache.find(key);
    if (it != s_gemm_cache.end()) return it->second;

    GemmPlan plan;
    plan.valid = false;

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&plan.matmul_desc, CUBLAS_COMPUTE_16F, CUDA_R_16F));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.matmul_desc,
        CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.matmul_desc,
        CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    if (has_bias) {
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.matmul_desc,
            CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    }

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.layout_A, CUDA_R_16F,
        opA == CUBLAS_OP_N ? col_m : col_k,
        opA == CUBLAS_OP_N ? col_k : col_m, ldA));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.layout_B, CUDA_R_16F,
        opB == CUBLAS_OP_N ? col_k : col_n,
        opB == CUBLAS_OP_N ? col_n : col_k, ldB));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.layout_C, CUDA_R_16F, col_m, col_n, ldC));

    cublasLtMatmulPreference_t pref;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &s_workspace_size, sizeof(s_workspace_size)));

    cublasLtMatmulHeuristicResult_t results[8];
    int n_results = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(s_cublaslt, plan.matmul_desc,
        plan.layout_A, plan.layout_B, plan.layout_C, plan.layout_C,
        pref, 8, results, &n_results));
    cublasLtMatmulPreferenceDestroy(pref);

    if (n_results > 0) {
        plan.algo = results[0].algo;
        plan.valid = true;
    }

    return s_gemm_cache.emplace(key, plan).first->second;
}

static void lt_gemm(cudaStream_t stream,
                     cublasOperation_t opA, cublasOperation_t opB,
                     int col_m, int col_n, int col_k,
                     const half* A, int ldA,
                     const half* B, int ldB,
                     half* C, int ldC,
                     bool accumulate, const half* bias = nullptr) {
    half alpha = __float2half(1.0f);
    half beta = __float2half(accumulate ? 1.0f : 0.0f);

    GemmPlan& plan = get_gemm_plan(col_m, col_n, col_k, ldA, ldB, ldC,
                                    opA, opB, accumulate, bias != nullptr);
    if (plan.valid) {
        if (bias) {
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.matmul_desc,
                CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
        }
        CUBLAS_CHECK(cublasLtMatmul(s_cublaslt, plan.matmul_desc,
            &alpha, A, plan.layout_A, B, plan.layout_B,
            &beta, C, plan.layout_C, C, plan.layout_C,
            &plan.algo, s_workspace, s_workspace_size, stream));
    } else {
        CUBLAS_CHECK(cublasGemmEx(s_cublas, opA, opB,
            col_m, col_n, col_k, &alpha, A, CUDA_R_16F, ldA,
            B, CUDA_R_16F, ldB, &beta, C, CUDA_R_16F, ldC,
            CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
}

// =========================================================================
// gemm.h implementation
// =========================================================================

void gemm_nn(cudaStream_t stream, const half* X, int m, int k,
             const half* W, int n, half* Y) {
    lt_gemm(stream, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, W, n, X, k, Y, n, false);
}

void gemm_nn_bias(cudaStream_t stream, const half* X, int m, int k,
                  const half* W, int n, const half* bias, half* Y) {
    lt_gemm(stream, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, W, n, X, k, Y, n, false, bias);
}

void gemm_nt(cudaStream_t stream, const half* X, int m, int k,
             const half* W, int n, half* Y) {
    lt_gemm(stream, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, W, k, X, k, Y, n, false);
}

void gemm_nt_bias(cudaStream_t stream, const half* X, int m, int k,
                  const half* W, int n, const half* bias, half* Y) {
    lt_gemm(stream, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, W, k, X, k, Y, n, false, bias);
}

void batched_gemm_nn(cudaStream_t stream,
                     const half* A, const half* B, half* C,
                     int batch, int m, int n, int k,
                     long long strideA, long long strideB, long long strideC) {
    half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(s_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k, &alpha, B, CUDA_R_16F, n, strideB,
        A, CUDA_R_16F, k, strideA, &beta, C, CUDA_R_16F, n, strideC,
        batch, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void batched_gemm_nt(cudaStream_t stream,
                     const half* A, const half* B, half* C,
                     int batch, int m, int n, int k,
                     long long strideA, long long strideB, long long strideC) {
    half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k, &alpha, B, CUDA_R_16F, k, strideB,
        A, CUDA_R_16F, k, strideA, &beta, C, CUDA_R_16F, n, strideC,
        batch, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void batched_gemm_nt_ex(cudaStream_t stream,
                        const half* A, int ldA, long long strideA,
                        const half* B, int ldB, long long strideB,
                        half* C, int ldC, long long strideC,
                        int batch, int m, int n, int k) {
    half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    // Row-major NT: C[b,m,n] = A[b,m,k] @ B[b,n,k]^T
    // Col-major: C'[n,m] = B_T'[n,k] @ A'[k,m]
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k, &alpha, B, CUDA_R_16F, ldB, strideB,
        A, CUDA_R_16F, ldA, strideA, &beta, C, CUDA_R_16F, ldC, strideC,
        batch, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
