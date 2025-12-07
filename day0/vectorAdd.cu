#include <iostream>
#include <math.h>
 
__global__
void add(int n, float *a, float *b, float *c)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride)
        c[i] = a[i] + b[i];
}
 
int main(void)
{
     int N = 1<<20;
    //float *x = new float[N];
    //float *y = new float[N];
    float *A = nullptr;
    float *B = nullptr;
    float *C = nullptr;

    float *devA = nullptr;
    float *devB = nullptr;
    float *devC = nullptr;
 
    cudaMallocHost(&A, N*sizeof(float));
    cudaMallocHost(&B, N*sizeof(float));
    cudaMallocHost(&C, N*sizeof(float));
 
    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    cudaMalloc(&devA, N*sizeof(float));
    cudaMalloc(&devB, N*sizeof(float));
    cudaMalloc(&devC, N*sizeof(float));

    cudaMemcpy(devA, A, N*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(devB, B, N*sizeof(float), cudaMemcpyDefault);
    cudaMemset(devC, 0, N*sizeof(float));
 
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, devA, devB, devC);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Launch error: " << cudaGetErrorString(err) << std::endl;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Sync error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(C, devC, N*sizeof(float), cudaMemcpyDefault);
 
    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(C[i]-3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;
 
 // Free memory
 cudaFreeHost(A);
 cudaFreeHost(B);
 cudaFreeHost(C);
 cudaFree(devA);
 cudaFree(devB);
 cudaFree(devC);
 //   delete [] x;
 //   delete [] y;
  return 0;
}
