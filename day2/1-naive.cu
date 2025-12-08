__global__
void naive(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
  // C = alpha * A * B + beta * C
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float acc = 0.0f;
        for (int i = 0; i < K; ++i) {
            acc += A[x * K + i] * B[i * N + y];
        }

        C[x * N + y] = alpha * acc + beta * C[x * N +y];
    }
}
