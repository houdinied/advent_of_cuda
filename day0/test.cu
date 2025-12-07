// test.cu
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add(float* a, float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum = a[idx] + b[idx];
        for (int i = 0; i < 1000; ++i)   // force real work so it can't run on CPU
            sum += 0.000001f * i;
        c[idx] = sum;
    }
}

int main() {
    const int N = 1 << 26;               // 67 million elements
    float *a, *b, *c;

    cudaMallocManaged(&a, N * sizeof(float));
    cudaMallocManaged(&b, N * sizeof(float));
    cudaMallocManaged(&c, N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Option 1 – Simple and works on all CUDA versions (recommended for now)
    int dev = 0;                                   // use GPU 0 (your 4090)
    cudaMemPrefetchAsync(a, N*sizeof(float), dev);
    cudaMemPrefetchAsync(b, N*sizeof(float), dev);
    cudaMemPrefetchAsync(c, N*sizeof(float), dev);

    // Option 2 – If you have CUDA ≥ 11.7 and want the newer syntax, use this instead:
    // cudaMemPrefetchAsync(a, N*sizeof(float), cudaMemLocationTypeDevice, dev);
    // (just uncomment the three lines above and comment the three simple ones)

    add<<< (N + 255)/256, 256 >>>(a, b, c, N);

    cudaDeviceSynchronize();   // waits for the kernel to finish

    printf("c[0] = %f (should be very close to 3.0)\n", c[0]);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}
