// bench_fp8.cu — FP8 vs FP16 microbenchmark for all key encoder GEMM shapes
//
// Compares:
//   - FP16 cublasLt (baseline)
//   - FP8 cublasLt  (per-tensor scaling via A/B scale pointers)
//   - FP8 CUTLASS SM120 NN (blockwise scaling, 128x128x128 tile)
//   - FP8 CUTLASS SM120 NT (blockwise scaling, 128x128x128 tile)
//   - FP8 quantize-only overhead
//
// Build: make bench_fp8
// Usage: ./bench_fp8

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include "cutlass_fp8_gemm.h"
#include "kernels_fp8.h"

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s = (x); if (s) { \
    fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s); exit(1); } } while(0)

// ---------------------------------------------------------------------------
// Timer
// ---------------------------------------------------------------------------
struct Timer {
    cudaEvent_t start, stop;
    cudaStream_t stream;
    Timer(cudaStream_t s) : stream(s) {
        cudaEventCreate(&start); cudaEventCreate(&stop);
    }
    ~Timer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin() { cudaEventRecord(start, stream); }
    float end_us() {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        return ms * 1000.0f;
    }
};

// ---------------------------------------------------------------------------
// FP16 cublasLt GEMM (NN row-major: Y[M,N] = X[M,K] @ W[K,N])
// ---------------------------------------------------------------------------
static cublasLtHandle_t g_lt;
static void* g_ws;
static size_t g_ws_size = 32 * 1024 * 1024;

static float bench_fp16_cublas(cudaStream_t stream, Timer& t,
                                 int M, int N, int K,
                                 const half* X, const half* W, half* Y,
                                 int warmup, int iters) {
    // col-major: Y^T[N,M] = W^T[N,K] @ X^T[K,M]  (NN)
    __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    auto opA = CUBLAS_OP_N, opB = CUBLAS_OP_N;
    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_16F, CUDA_R_16F);
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));
    cublasLtMatrixLayout_t lA, lB, lC;
    // col-major: A is W^T[N,K], B is X^T[K,M], C is Y^T[N,M]
    cublasLtMatrixLayoutCreate(&lA, CUDA_R_16F, N, K, N);
    cublasLtMatrixLayoutCreate(&lB, CUDA_R_16F, K, M, K);
    cublasLtMatrixLayoutCreate(&lC, CUDA_R_16F, N, M, N);
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &g_ws_size, sizeof(g_ws_size));
    cublasLtMatmulHeuristicResult_t result;
    int returned;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(g_lt, desc, lA, lB, lC, lC,
                                                  pref, 1, &result, &returned));

    auto run = [&]() {
        cublasLtMatmul(g_lt, desc, &alpha, W, lA, X, lB, &beta, Y, lC, Y, lC,
                       &result.algo, g_ws, g_ws_size, stream);
    };
    for (int i = 0; i < warmup; i++) run();
    cudaStreamSynchronize(stream);
    t.begin();
    for (int i = 0; i < iters; i++) run();
    float us = t.end_us() / iters;

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(lA); cublasLtMatrixLayoutDestroy(lB); cublasLtMatrixLayoutDestroy(lC);
    cublasLtMatmulDescDestroy(desc);
    return us;
}

// ---------------------------------------------------------------------------
// FP8 cublasLt GEMM (NN): quantize X→FP8, then FP8 matmul
// ---------------------------------------------------------------------------
static float bench_fp8_cublas(cudaStream_t stream, Timer& t,
                                int M, int N, int K,
                                const half* X, const uint8_t* W_fp8,
                                const float* w_scale, float* a_scale,
                                uint8_t* act_buf, int* amax_buf,
                                half* Y,
                                int warmup, int iters) {
    // First calibrate to get a_scale
    quantize_absmax_fp16_to_fp8(X, act_buf, a_scale, M * K, amax_buf, stream);
    cudaStreamSynchronize(stream);

    // Build cublasLt FP8 plan
    // col-major layout: A=W^T[N,K], B=act^T[K,M], C=Y^T[N,M]
    auto opA = CUBLAS_OP_N, opB = CUBLAS_OP_N;
    cublasLtMatmulDesc_t desc;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
    const float* w_scale_ptr = w_scale;
    const float* a_scale_ptr = a_scale;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                 &w_scale_ptr, sizeof(w_scale_ptr)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                 &a_scale_ptr, sizeof(a_scale_ptr)));
    cublasLtMatrixLayout_t lA, lB, lC;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lA, CUDA_R_8F_E4M3, N, K, N));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lB, CUDA_R_8F_E4M3, K, M, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lC, CUDA_R_16F, N, M, N));
    cublasLtMatmulPreference_t pref;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                       &g_ws_size, sizeof(g_ws_size)));
    cublasLtMatmulHeuristicResult_t results[4];
    int n_results = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(g_lt, desc, lA, lB, lC, lC,
                                                         pref, 4, results, &n_results);
    cublasLtMatmulPreferenceDestroy(pref);

    if (st != CUBLAS_STATUS_SUCCESS || n_results == 0) {
        fprintf(stderr, "FP8 cublasLt: no algo for M=%d N=%d K=%d\n", M, N, K);
        cublasLtMatrixLayoutDestroy(lA); cublasLtMatrixLayoutDestroy(lB); cublasLtMatrixLayoutDestroy(lC);
        cublasLtMatmulDescDestroy(desc);
        return -1.0f;
    }
    auto algo = results[0].algo;

    float alpha = 1.0f, beta = 0.0f;
    auto run = [&]() {
        quantize_fp8_static(X, act_buf, a_scale, M * K, stream);
        cublasLtMatmul(g_lt, desc, &alpha, W_fp8, lA, act_buf, lB, &beta, Y, lC, Y, lC,
                       &algo, g_ws, g_ws_size, stream);
    };
    for (int i = 0; i < warmup; i++) run();
    cudaStreamSynchronize(stream);
    t.begin();
    for (int i = 0; i < iters; i++) run();
    float us = t.end_us() / iters;

    cublasLtMatrixLayoutDestroy(lA); cublasLtMatrixLayoutDestroy(lB); cublasLtMatrixLayoutDestroy(lC);
    cublasLtMatmulDescDestroy(desc);
    return us;
}

// ---------------------------------------------------------------------------
// FP8 CUTLASS SM120: quantize X→FP8, then CUTLASS GEMM
// W_fp8 must be [N,K] row-major (K contiguous)
// ---------------------------------------------------------------------------
static float bench_fp8_cutlass_nn(cudaStream_t stream, Timer& t,
                                    int M, int N, int K,
                                    const half* X, const uint8_t* W_fp8,
                                    float alpha,
                                    uint8_t* act_buf, float* a_scale,
                                    half* Y,
                                    int warmup, int iters) {
    // Validate
    {
        quantize_fp8_static(X, act_buf, a_scale, M * K, stream);
        int r = cutlass_fp8_nn(stream, M, N, K, act_buf, W_fp8, Y, alpha);
        if (r != 0) {
            fprintf(stderr, "CUTLASS FP8 NN: failed for M=%d N=%d K=%d (rc=%d)\n", M, N, K, r);
            return -1.0f;
        }
    }

    auto run = [&]() {
        quantize_fp8_static(X, act_buf, a_scale, M * K, stream);
        cutlass_fp8_nn(stream, M, N, K, act_buf, W_fp8, Y, alpha);
    };
    for (int i = 0; i < warmup; i++) run();
    cudaStreamSynchronize(stream);
    t.begin();
    for (int i = 0; i < iters; i++) run();
    return t.end_us() / iters;
}

// Alias for NT shapes — same kernel, W already [N,K] row-major
static float bench_fp8_cutlass_nt(cudaStream_t stream, Timer& t,
                                    int M, int N, int K,
                                    const half* X, const uint8_t* W_fp8,
                                    float alpha,
                                    uint8_t* act_buf, float* a_scale,
                                    half* Y,
                                    int warmup, int iters) {
    return bench_fp8_cutlass_nn(stream, t, M, N, K, X, W_fp8, alpha,
                                 act_buf, a_scale, Y, warmup, iters);
}

// ---------------------------------------------------------------------------
// Quantize only (to measure its overhead separately)
// ---------------------------------------------------------------------------
static float bench_quantize_only(cudaStream_t stream, Timer& t,
                                   int M, int K,
                                   const half* X, uint8_t* act_buf, float* a_scale,
                                   int warmup, int iters) {
    for (int i = 0; i < warmup; i++)
        quantize_fp8_static(X, act_buf, a_scale, M * K, stream);
    cudaStreamSynchronize(stream);
    t.begin();
    for (int i = 0; i < iters; i++)
        quantize_fp8_static(X, act_buf, a_scale, M * K, stream);
    return t.end_us() / iters;
}

// ---------------------------------------------------------------------------
// Standalone correctness test: known-value FP8 GEMM vs CPU reference
// ---------------------------------------------------------------------------
static bool test_fp8_correctness(cudaStream_t stream) {
    printf("\n=== FP8 CUTLASS Correctness Test ===\n");
    bool all_pass = true;

    struct TestCase { int M, N, K; const char* label; };
    TestCase cases[] = {
        {128,  128,  128,  "single_tile"},
        {256,  1024, 1024, "K1024"},
        {128,  1024, 4096, "K4096"},
        {64,   4096, 1024, "FF_lin1_small"},
        {870,  1024, 4096, "FF_lin2_large"},
    };

    for (auto& tc : cases) {
        int M = tc.M, N = tc.N, K = tc.K;

        // Allocate host buffers
        std::vector<float> h_A(M * K), h_B(K * N);
        std::vector<uint8_t> h_A_fp8(M * K), h_B_fp8(K * N);
        std::vector<float> h_C_ref(M * N, 0.0f);

        // Fill with small integer values that fit in E4M3 exactly
        // E4M3 can represent integers 0-8 exactly
        srand(42);
        for (int i = 0; i < M * K; i++) {
            float v = (float)(rand() % 5);  // 0..4
            h_A[i] = v;
            // E4M3: sign(1) exp(4) mantissa(3), bias=7
            // Cast to fp8 e4m3: for small ints this is exact
            __nv_fp8_e4m3 fp8_val = __nv_fp8_e4m3(v);
            memcpy(&h_A_fp8[i], &fp8_val, 1);
        }
        for (int i = 0; i < K * N; i++) {
            float v = (float)(rand() % 5);
            h_B[i] = v;
            __nv_fp8_e4m3 fp8_val = __nv_fp8_e4m3(v);
            memcpy(&h_B_fp8[i], &fp8_val, 1);
        }

        // CPU reference: C = A @ B (A is [M,K] row-major, B is [K,N] row-major)
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float acc = 0;
                for (int kk = 0; kk < K; kk++)
                    acc += h_A[m * K + kk] * h_B[kk * N + n];
                h_C_ref[m * N + n] = acc;
            }
        }

        // Transpose B from [K,N] to [N,K] row-major (CUTLASS expects [N,K])
        std::vector<uint8_t> h_B_T(N * K);
        for (int r = 0; r < K; r++)
            for (int c = 0; c < N; c++)
                h_B_T[c * K + r] = h_B_fp8[r * N + c];

        // Upload to GPU
        uint8_t *d_A, *d_B;
        half *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, M * K));
        CUDA_CHECK(cudaMalloc(&d_B, N * K));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A_fp8.data(), M * K, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B_T.data(), N * K, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(half)));

        // Run CUTLASS FP8 (B is [N,K] row-major)
        int rc = cutlass_fp8_nn(stream, M, N, K, d_A, d_B, d_C, 1.0f);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (rc != 0) {
            printf("  %-16s: FAILED (launch error rc=%d)\n", tc.label, rc);
            all_pass = false;
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
            continue;
        }

        // Download and compare
        std::vector<half> h_C(M * N);
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost));

        int errors = 0;
        float max_rel_err = 0;
        for (int i = 0; i < M * N; i++) {
            float got = __half2float(h_C[i]);
            float ref = h_C_ref[i];
            float abs_err = fabsf(got - ref);
            float rel_err = (ref != 0) ? abs_err / fabsf(ref) : abs_err;
            if (rel_err > max_rel_err) max_rel_err = rel_err;
            // FP8 E4M3 accumulation may have rounding; use generous threshold
            if (abs_err > 1.0f && rel_err > 0.05f) {
                if (errors < 3) {
                    printf("    [%d,%d] got=%.2f ref=%.2f err=%.4f\n",
                           i / N, i % N, got, ref, abs_err);
                }
                errors++;
            }
        }

        bool pass = (errors == 0);
        printf("  %-16s M=%3d N=%4d K=%4d: %s (max_rel=%.4f, errors=%d/%d)\n",
               tc.label, M, N, K,
               pass ? "PASS" : "FAIL", max_rel_err, errors, M * N);
        if (!pass) all_pass = false;

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    return all_pass;
}

// ---------------------------------------------------------------------------
// Compare CUTLASS FP8 vs cublasLt FP8 with random float data
// ---------------------------------------------------------------------------
static void test_cutlass_vs_cublas(cudaStream_t stream) {
    printf("\n=== CUTLASS FP8 vs cublasLt FP8 (random float data) ===\n");

    struct TestCase { int M, N, K; const char* label; };
    TestCase cases[] = {
        {131,  4096, 1024, "ff1_w1"},
        {131,  1024, 4096, "ff1_w2"},
        {131,  3072, 1024, "qkv"},
        {131,  1024, 1024, "out_w"},
    };

    for (auto& tc : cases) {
        int M = tc.M, N = tc.N, K = tc.K;

        // Generate random FP16 values on host
        std::vector<half> h_X(M * K), h_W(K * N);
        srand(123);
        for (int i = 0; i < M * K; i++)
            h_X[i] = __float2half(((float)(rand() % 2000) - 1000) / 100.0f);  // [-10, 10]
        for (int i = 0; i < K * N; i++)
            h_W[i] = __float2half(((float)(rand() % 2000) - 1000) / 1000.0f);  // [-1, 1]

        // Upload FP16 data
        half *d_X, *d_W;
        CUDA_CHECK(cudaMalloc(&d_X, M * K * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_W, K * N * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));

        // Quantize weight to FP8 (NN: [K,N] → same layout)
        uint8_t *d_W_fp8;
        float *d_w_scale, *d_a_scale;
        int *d_amax;
        uint8_t *d_act_buf;
        CUDA_CHECK(cudaMalloc(&d_W_fp8, K * N));
        CUDA_CHECK(cudaMalloc(&d_w_scale, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_a_scale, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_amax, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_act_buf, M * K));

        quantize_absmax_fp16_to_fp8(d_W, (uint8_t*)d_W_fp8, d_w_scale, K * N, d_amax, stream);
        quantize_absmax_fp16_to_fp8(d_X, d_act_buf, d_a_scale, M * K, d_amax, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Read scales
        float h_w_scale, h_a_scale;
        CUDA_CHECK(cudaMemcpy(&h_w_scale, d_w_scale, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_a_scale, d_a_scale, sizeof(float), cudaMemcpyDeviceToHost));

        // --- cublasLt FP8 ---
        half *d_Y_cublas;
        CUDA_CHECK(cudaMalloc(&d_Y_cublas, M * N * sizeof(half)));
        {
            auto opA = CUBLAS_OP_N, opB = CUBLAS_OP_N;
            cublasLtMatmulDesc_t desc;
            CUBLAS_CHECK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
            const float* ws_ptr = d_w_scale;
            const float* as_ptr = d_a_scale;
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &ws_ptr, sizeof(ws_ptr)));
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &as_ptr, sizeof(as_ptr)));
            cublasLtMatrixLayout_t lA, lB, lC;
            CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lA, CUDA_R_8F_E4M3, N, K, N));
            CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lB, CUDA_R_8F_E4M3, K, M, K));
            CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lC, CUDA_R_16F, N, M, N));
            cublasLtMatmulPreference_t pref;
            CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
            CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_ws_size, sizeof(g_ws_size)));
            cublasLtMatmulHeuristicResult_t result;
            int n_results = 0;
            CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(g_lt, desc, lA, lB, lC, lC, pref, 1, &result, &n_results));
            float alpha = 1.0f, beta = 0.0f;
            CUBLAS_CHECK(cublasLtMatmul(g_lt, desc, &alpha, d_W_fp8, lA, d_act_buf, lB, &beta, d_Y_cublas, lC, d_Y_cublas, lC,
                                         &result.algo, g_ws, g_ws_size, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            cublasLtMatmulPreferenceDestroy(pref);
            cublasLtMatrixLayoutDestroy(lA); cublasLtMatrixLayoutDestroy(lB); cublasLtMatrixLayoutDestroy(lC);
            cublasLtMatmulDescDestroy(desc);
        }

        // --- CUTLASS FP8 ---
        // Transpose W_fp8 from [K,N] to [N,K] for CUTLASS
        uint8_t *d_W_fp8_T;
        CUDA_CHECK(cudaMalloc(&d_W_fp8_T, N * K));
        {
            // Use the transpose_u8 kernel
            void* tmp;
            CUDA_CHECK(cudaMalloc(&tmp, K * N));
            CUDA_CHECK(cudaMemcpy(tmp, d_W_fp8, K * N, cudaMemcpyDeviceToDevice));
            // Simple host transpose
            std::vector<uint8_t> h_w(K * N), h_wt(N * K);
            CUDA_CHECK(cudaMemcpy(h_w.data(), d_W_fp8, K * N, cudaMemcpyDeviceToHost));
            for (int r = 0; r < K; r++)
                for (int c = 0; c < N; c++)
                    h_wt[c * K + r] = h_w[r * N + c];
            CUDA_CHECK(cudaMemcpy(d_W_fp8_T, h_wt.data(), N * K, cudaMemcpyHostToDevice));
            cudaFree(tmp);
        }

        half *d_Y_cutlass;
        CUDA_CHECK(cudaMalloc(&d_Y_cutlass, M * N * sizeof(half)));
        float cutlass_alpha = h_a_scale * h_w_scale;
        cutlass_fp8_nn(stream, M, N, K, d_act_buf, d_W_fp8_T, d_Y_cutlass, cutlass_alpha);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // --- Compare ---
        std::vector<half> h_cublas(M * N), h_cutlass(M * N);
        CUDA_CHECK(cudaMemcpy(h_cublas.data(), d_Y_cublas, M * N * sizeof(half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_cutlass.data(), d_Y_cutlass, M * N * sizeof(half), cudaMemcpyDeviceToHost));

        double sum_sq = 0, max_abs = 0, sum_cublas_sq = 0;
        int n_out = M * N;
        for (int i = 0; i < n_out; i++) {
            double cv = __half2float(h_cublas[i]);
            double xv = __half2float(h_cutlass[i]);
            double diff = xv - cv;
            sum_sq += diff * diff;
            sum_cublas_sq += cv * cv;
            if (fabs(diff) > max_abs) max_abs = fabs(diff);
        }
        double rms_err = sqrt(sum_sq / n_out);
        double rms_cublas = sqrt(sum_cublas_sq / n_out);
        printf("  %-10s M=%3d N=%4d K=%4d: rms_cublas=%.3f rms_err=%.3f max_abs=%.3f ratio=%.6f\n",
               tc.label, M, N, K, rms_cublas, rms_err, max_abs, rms_err / (rms_cublas + 1e-10));
        // Print first 4 values
        printf("    cublas: ");
        for (int i = 0; i < 4; i++) printf("%10.3f", __half2float(h_cublas[i]));
        printf("\n    cutlass:");
        for (int i = 0; i < 4; i++) printf("%10.3f", __half2float(h_cutlass[i]));
        printf("\n");

        cudaFree(d_X); cudaFree(d_W); cudaFree(d_W_fp8); cudaFree(d_W_fp8_T);
        cudaFree(d_w_scale); cudaFree(d_a_scale); cudaFree(d_amax); cudaFree(d_act_buf);
        cudaFree(d_Y_cublas); cudaFree(d_Y_cutlass);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasLtCreate(&g_lt));
    CUDA_CHECK(cudaMalloc(&g_ws, g_ws_size));

    // Initialize CUTLASS FP8 module (scale buffers + workspace)
    cutlass_fp8_init(870, 4096, 4096, stream);

    // Run correctness test first
    bool correct = test_fp8_correctness(stream);
    if (!correct) {
        fprintf(stderr, "\nFP8 correctness test FAILED — skipping benchmarks\n");
        cutlass_fp8_free();
        cudaFree(g_ws);
        cublasLtDestroy(g_lt);
        cudaStreamDestroy(stream);
        return 1;
    }

    // Compare CUTLASS vs cublasLt
    test_cutlass_vs_cublas(stream);

    // Allocate buffers (enough for largest shape: 870×4096)
    size_t max_elems = 870 * 4096;
    half*    buf_fp16   = nullptr;
    uint8_t* buf_fp8    = nullptr;
    uint8_t* act_buf    = nullptr;
    half*    out_buf    = nullptr;
    float*   w_scale    = nullptr;
    float*   a_scale    = nullptr;
    int*     amax_buf   = nullptr;

    CUDA_CHECK(cudaMalloc(&buf_fp16, max_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&buf_fp8,  max_elems * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&act_buf,  max_elems * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&out_buf,  870 * 4096 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&w_scale,  sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a_scale,  sizeof(float)));
    CUDA_CHECK(cudaMalloc(&amax_buf, sizeof(int)));

    // Fill FP16 buffer with ~1.0 values
    CUDA_CHECK(cudaMemset(buf_fp16, 0x3C, max_elems * sizeof(half)));
    // Fill FP8 weight with reasonable values
    CUDA_CHECK(cudaMemset(buf_fp8,  0x3C, max_elems * sizeof(uint8_t)));

    // Compute weight scale from absmax
    quantize_absmax_fp16_to_fp8(buf_fp16, buf_fp8, w_scale, 1024 * 4096, amax_buf, stream);
    // Compute act scale
    quantize_absmax_fp16_to_fp8(buf_fp16, act_buf, a_scale, 870 * 1024, amax_buf, stream);
    cudaStreamSynchronize(stream);
    float host_alpha = 1.0f;  // approximate; real alpha = act_scale * wt_scale

    Timer t(stream);
    int warmup = 20, iters = 200;

    // --- NN shapes (W stored [K,N] row-major) ---
    printf("\n=== NN shapes (Y = X @ W, W[K,N] row-major) ===\n");
    struct Shape { const char* label; int M, N, K; };
    Shape nn_shapes[] = {
        // FF linear1: Y[T,4096] = X[T,1024] @ W[1024,4096]
        {"FF_lin1",  64,  4096, 1024},
        {"FF_lin1", 128,  4096, 1024},
        {"FF_lin1", 256,  4096, 1024},
        {"FF_lin1", 512,  4096, 1024},
        {"FF_lin1", 870,  4096, 1024},
        // FF linear2: Y[T,1024] = X[T,4096] @ W[4096,1024]
        {"FF_lin2",  64,  1024, 4096},
        {"FF_lin2", 128,  1024, 4096},
        {"FF_lin2", 256,  1024, 4096},
        {"FF_lin2", 512,  1024, 4096},
        {"FF_lin2", 870,  1024, 4096},
        // QKV: Y[T,3072] = X[T,1024] @ W[1024,3072]
        {"QKV",      64,  3072, 1024},
        {"QKV",     128,  3072, 1024},
        {"QKV",     256,  3072, 1024},
        {"QKV",     512,  3072, 1024},
    };

    printf("\n%-14s  %4s  %4s  %4s  %8s  %9s  %11s  %8s  %7s  %7s\n",
           "Shape", "M", "N", "K",
           "FP16(us)", "FP8Lt(us)", "CUTLASS(us)", "Quant(us)",
           "Lt/FP16", "CUT/FP16");
    printf("%-14s  %4s  %4s  %4s  %8s  %9s  %11s  %8s  %7s  %7s\n",
           "--------------", "----", "----", "----",
           "--------", "---------", "-----------", "--------",
           "-------", "-------");

    auto pct = [](float v, float base) -> const char* {
        static char buf[8][16]; static int idx = 0; idx = (idx+1)%8;
        if (v < 0) { snprintf(buf[idx], 16, "   n/a "); return buf[idx]; }
        snprintf(buf[idx], 16, v < base * 0.99f ? " %5.2fx*" : " %5.2fx ",
                 base / v);
        return buf[idx];
    };

    for (auto& s : nn_shapes) {
        int M = s.M, N = s.N, K = s.K;
        const half* X = buf_fp16;
        const uint8_t* W_fp8 = buf_fp8;  // uniform fill, layout doesn't matter for perf

        float fp16_us = bench_fp16_cublas(stream, t, M, N, K, X, buf_fp16, out_buf, warmup, iters);
        float fp8lt_us = bench_fp8_cublas(stream, t, M, N, K, X, buf_fp8, w_scale, a_scale,
                                           act_buf, amax_buf, out_buf, warmup, iters);
        float cutlass_us = bench_fp8_cutlass_nn(stream, t, M, N, K, X, W_fp8, host_alpha,
                                                  act_buf, a_scale, out_buf, warmup, iters);
        float quant_us = bench_quantize_only(stream, t, M, K, X, act_buf, a_scale, warmup, iters);

        printf("%-14s  %4d  %4d  %4d  %8.1f  %9.1f  %11.1f  %8.1f  %s  %s\n",
               s.label, M, N, K,
               fp16_us, fp8lt_us, cutlass_us, quant_us,
               pct(fp8lt_us, fp16_us), pct(cutlass_us, fp16_us));
    }

    // --- NT shapes (W stored [N,K] row-major) ---
    printf("\n=== NT shapes (Y = X @ W^T, W[N,K] row-major) ===\n");
    Shape nt_shapes[] = {
        // conv_pw1: Y[T,2048] = X[T,1024] @ W[2048,1024]^T
        {"conv_pw1",  64,  2048, 1024},
        {"conv_pw1", 128,  2048, 1024},
        {"conv_pw1", 256,  2048, 1024},
        {"conv_pw1", 512,  2048, 1024},
        {"conv_pw1", 870,  2048, 1024},
        // conv_pw2: Y[T,1024] = X[T,1024] @ W[1024,1024]^T
        {"conv_pw2",  64,  1024, 1024},
        {"conv_pw2", 128,  1024, 1024},
        {"conv_pw2", 256,  1024, 1024},
        {"conv_pw2", 512,  1024, 1024},
        {"conv_pw2", 870,  1024, 1024},
    };

    printf("\n%-14s  %4s  %4s  %4s  %8s  %11s  %8s\n",
           "Shape", "M", "N", "K",
           "FP16(us)", "CUTLASS(us)", "CUT/FP16");
    printf("%-14s  %4s  %4s  %4s  %8s  %11s  %8s\n",
           "--------------", "----", "----", "----",
           "--------", "-----------", "--------");

    for (auto& s : nt_shapes) {
        int M = s.M, N = s.N, K = s.K;
        const half* X = buf_fp16;
        const uint8_t* W_fp8 = buf_fp8;

        float fp16_us = bench_fp16_cublas(stream, t, M, N, K, X, buf_fp16, out_buf, warmup, iters);
        float cutlass_us = bench_fp8_cutlass_nt(stream, t, M, N, K, X, W_fp8, host_alpha,
                                                  act_buf, a_scale, out_buf, warmup, iters);

        printf("%-14s  %4d  %4d  %4d  %8.1f  %11.1f  %s\n",
               s.label, M, N, K,
               fp16_us, cutlass_us,
               pct(cutlass_us, fp16_us));
    }

    // Cleanup
    cutlass_fp8_free();
    cudaFree(buf_fp16); cudaFree(buf_fp8); cudaFree(act_buf);
    cudaFree(out_buf); cudaFree(w_scale); cudaFree(a_scale);
    cudaFree(amax_buf);
    cudaFree(g_ws);
    cublasLtDestroy(g_lt);
    cudaStreamDestroy(stream);
    return 0;
}
