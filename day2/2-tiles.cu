__global__
void tiled_blocks(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // Shared memory tiles with +1 padding to avoid bank conflicts
    __shared__ float tileA[32][33];
    __shared__ float tileB[32][33];

    // Thread's position in output matrix C
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;  // Which row of C
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;  // Which column of C

    float acc = 0.0f;

    // Loop over tiles along the K dimension
    // We need K/32 tiles to cover the entire dot product
    for (int tileIdx = 0; tileIdx < (K + 31) / 32; ++tileIdx) {

        // A is M×K, we want row 'row' and columns [tileIdx*32 : tileIdx*32+32]
        int aCol = tileIdx * 32 + threadIdx.x;  // Column in A (along K dimension)
        if (row < M && aCol < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;  // Padding for out-of-bounds
        }

        // B is K×N, we want rows [tileIdx*32 : tileIdx*32+32] and column 'col'
        int bRow = tileIdx * 32 + threadIdx.y;  // Row in B (along K dimension)
        if (bRow < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;  // Padding for out-of-bounds
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}
