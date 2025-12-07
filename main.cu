#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>

#ifndef M_SIZE
#define M_SIZE 4096
#endif
#ifndef N_SIZE
#define N_SIZE 4096
#endif
#ifndef K_SIZE
#define K_SIZE 4096
#endif

using DataType = float;

void check(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

// Naive kernel – replace this every day
__global__ void my_sgemm(int M, int N, int K,
                         const DataType* A, const DataType* B, DataType* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(int argc, char** argv) {
    int M = M_SIZE, N = N_SIZE, K = K_SIZE;
    if (argc == 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }

    printf("GEMM %d × %d × %d\n", M, N, K);

    size_t sizeA = M * K * sizeof(DataType);
    size_t sizeB = K * N * sizeof(DataType);
    size_t sizeC = M * N * sizeof(DataType);

    DataType *h_A = (DataType*)malloc(sizeA);
    DataType *h_B = (DataType*)malloc(sizeB);
    DataType *h_C_ref  = (DataType*)malloc(sizeC);
    DataType *h_C_mine = (DataType*)malloc(sizeC);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < M*K; ++i) h_A[i] = dis(gen);
    for (size_t i = 0; i < K*N; ++i) h_B[i] = dis(gen);

    DataType *d_A, *d_B, *d_C_ref, *d_C_mine;
    check(cudaMalloc(&d_A, sizeA));
    check(cudaMalloc(&d_B, sizeB));
    check(cudaMalloc(&d_C_ref, sizeC));
    check(cudaMalloc(&d_C_mine, sizeC));

    check(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    dim3 block(32, 8);
    dim3 grid((N + block.x - 1)/block.x, (M + block.y - 1)/block.y);

    // Warm-up
    my_sgemm<<<grid, block>>>(M, N, K, d_A, d_B, d_C_mine);
    check(cudaDeviceSynchronize());

    // cuBLAS reference
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    auto t1 = std::chrono::high_resolution_clock::now();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_ref, N);
    check(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();
    double cublas_ms = std::chrono::duration<double,std::milli>(t2-t1).count();
    printf("cuBLAS :  %.2f ms  →  %.2f TFLOPS\n", cublas_ms, 2.0*M*N*K/cublas_ms/1e9);
    check(cudaMemcpy(h_C_ref, d_C_ref, sizeC, cudaMemcpyDeviceToHost));

    // Your kernel timing (10 runs average)
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 10; ++i)
        my_sgemm<<<grid, block>>>(M, N, K, d_A, d_B, d_C_mine);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 10.0f;
    printf("Yours   :  %.2f ms  →  %.2f TFLOPS\n", ms, 2.0*M*N*K/ms/1e9);

    // Correctness – FIXED PART
    check(cudaMemcpy(h_C_mine, d_C_mine, sizeC, cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    double sum_err = 0.0;
    long long total = (long long)M * N;

    for (long long i = 0; i < total; ++i) {
        double err = fabs(h_C_ref[i] - h_C_mine[i]);
        sum_err += err;
        if (err > max_err) max_err = err;
    }
    double avg_err = sum_err / total;

    printf("Error   :  max = %.3e   avg = %.3e  →  %s\n",
           max_err, avg_err,
           (max_err < 1e-3 ? "PASSED" : "FAILED"));

    // Cleanup
    cublasDestroy(handle);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_ref); cudaFree(d_C_mine);
    free(h_A); free(h_B); free(h_C_ref); free(h_C_mine);

    return 0;
}
