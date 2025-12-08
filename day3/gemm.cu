#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <random>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include "1-wmma.hu"

inline void cudaCheckError(cudaError_t error_code, const char *file, int line) {
    if (error_code != cudaSuccess) {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n",
                error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}

#define cudaCheck(err) (cudaCheckError(err, __FILE__, __LINE__))

void randomize_matrix(float *mat, int size) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < size; i++) {
        mat[i] = dist(gen);
    }
}

void randomize_matrix_half(__half *mat, int size) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < size; i++) {
        mat[i] = __float2half(dist(gen));
    }
}

bool verify_matrix(const float *mat_ref, const float *mat, int size) {
    float epsilon = 1e-3;
    for (int i = 0; i < size; i++) {
        if (std::abs(mat_ref[i] - mat[i]) > epsilon) {
            printf("Mismatch at index %d: ref=%f, got=%f\n", i, mat_ref[i], mat[i]);
            return false;
        }
    }
    return true;
}

void run_kernel(int kernel_num, int M, int N, int K, float alpha, const __half *A, const __half *B, float beta, float *C, int val, cublasHandle_t handle = nullptr) {

    if (kernel_num == 0) {
        // FP16 inputs with FP32 accumulation
        if (val) printf("cuBLAS FP16 with FP32 accumulate\n");
        cublasGemmEx(handle,
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K,
                     &alpha,
                     B, CUDA_R_16F, N,
                     A, CUDA_R_16F, K,
                     &beta,
                     C, CUDA_R_32F, N,
                     CUDA_R_32F,
                     CUBLAS_GEMM_DEFAULT);
    }
    else if (kernel_num == 1) {
        if (val) printf("Basic WMMA 16×16×16 fragments kernel\n");

        // For WMMA: 128 threads = 4 warps per block
        // 4 warps arranged as 2×2, each block covers 32×32 output
        dim3 blockDim(128, 1);
        dim3 gridDim((N + 31) / 32, (M + 31) / 32);
        //dim3 blockDim(32, 1);  Means: 32 threads in x, 1 thread in y, 1 in z
        wmma_gemm<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
        cudaDeviceSynchronize(); // Flush printf from kernel
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel_number> [kernel_number...]\n";
        std::cerr << "  kernel 0: cuBLAS FP16 with FP32 accumulate (8192x8192)\n";
        exit(EXIT_FAILURE);
    }

    std::vector<int> kernel_nums;
    for (int i = 1; i < argc; i++) {
        int kernel_num = std::stoi(argv[i]);
        if (kernel_num < 0) {
            std::cerr << "Invalid kernel number: " << kernel_num << "\n";
            exit(EXIT_FAILURE);
        }
        kernel_nums.push_back(kernel_num);
    }

    printf("Running %d kernel(s)\n", (int)kernel_nums.size());

    cublasHandle_t handle;
    if (cublasCreate(&handle)) {
        std::cerr << "Create cublas handle error.\n";
        exit(EXIT_FAILURE);
    }

    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    std::vector<int> SIZES = {2048, 4096};
    int max_size = SIZES[SIZES.size() - 1];

    float alpha = 1.0f, beta = 0.0f; // C = A * B

    // Allocate host memory - A and B are FP16, C is FP32
    __half *A = (__half *)malloc(sizeof(__half) * max_size * max_size);
    __half *B = (__half *)malloc(sizeof(__half) * max_size * max_size);
    float *C = (float *)malloc(sizeof(float) * max_size * max_size);
    float *C_ref = (float *)malloc(sizeof(float) * max_size * max_size);

    // Initialize matrices
    randomize_matrix_half(A, max_size * max_size);
    randomize_matrix_half(B, max_size * max_size);
    randomize_matrix(C, max_size * max_size);

    __half *dA = nullptr, *dB = nullptr;
    float *dC = nullptr, *dC_ref = nullptr;
    cudaCheck(cudaMalloc((void **)&dA, sizeof(__half) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dB, sizeof(__half) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(float) * max_size * max_size));

    cudaCheck(cudaMemcpy(dA, A, sizeof(__half) * max_size * max_size,
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B, sizeof(__half) * max_size * max_size,
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size,
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * max_size * max_size,
                         cudaMemcpyHostToDevice));

    int repeat_times = 10;

    for (int kernel_num : kernel_nums) {
        printf("\n========== Kernel %d ==========\n", kernel_num);

        for (int size : SIZES) {
            int M = size, N = size, K = size;

            printf("\n--- Size: %d x %d x %d ---\n", M, N, K);

            // Compute cuBLAS reference for this size
            cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * M * N, cudaMemcpyHostToDevice));
            run_kernel(0, M, N, K, alpha, dA, dB, beta, dC_ref, 0, handle);
            cudaCheck(cudaMemcpy(C_ref, dC_ref, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

            // Run the test kernel
            cudaCheck(cudaMemcpy(dC, C, sizeof(float) * M * N, cudaMemcpyHostToDevice));
            run_kernel(kernel_num, M, N, K, alpha, dA, dB, beta, dC, 1, handle);
            cudaCheck(cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));


            if (!verify_matrix(C_ref, C, M * N)) {
                printf("Verification FAILED\n");
                exit(EXIT_FAILURE);
            }
            printf("Verification PASSED\n");
           

            cudaEventRecord(beg);
            for (int j = 0; j < repeat_times; j++) {
                run_kernel(kernel_num, M, N, K, alpha, dA, dB, beta, dC, 0, handle);
            }
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&elapsed_time, beg, end);
            elapsed_time /= 1000.0; // Convert to seconds

            long flops = 2L * M * N * K;
            double time_spent = elapsed_time / repeat_times;
            double tflops = (repeat_times * flops * 1e-12) / elapsed_time;
            printf("Average time: %.6f s, Performance: %.2f TFLOPS\n", time_spent, tflops);
        }
    }

    free(A);
    free(B);
    free(C);
    free(C_ref);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dC_ref);
    cublasDestroy(handle);

    return 0;
}

