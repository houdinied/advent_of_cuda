__global__ void double_buffer(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    __shared__ float tileA[2][32][33];
    __shared__ float tileB[2][32][33];

    // Thread's position in output matrix C
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;  // Which row of C
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;  // Which column of C

    float acc = 0.0f;

    int writeIdx = 0;  // Buffer to write next tile into
    int readIdx = 1;   // Buffer to read/compute from

    int aCol = threadIdx.x;  // Columns 0-31 of A
    int bRow = threadIdx.y;  // Rows 0-31 of B

    if (row < M && aCol < K) {
        tileA[0][threadIdx.y][threadIdx.x] = A[row * K + aCol];
    } else {
        tileA[0][threadIdx.y][threadIdx.x] = 0.0f;
    }

    if (bRow < K && col < N) {
        tileB[0][threadIdx.y][threadIdx.x] = B[bRow * N + col];
    } else {
        tileB[0][threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    int numTiles = (K + 31) / 32;

    for (int tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
        // Swap buffers: what we just loaded becomes readable, prepare to write to other buffer
        readIdx = writeIdx;
        writeIdx = 1 - writeIdx;

        if (tileIdx + 1 < numTiles) {
            int nextTileIdx = tileIdx + 1;
            int aCol = nextTileIdx * 32 + threadIdx.x;
            int bRow = nextTileIdx * 32 + threadIdx.y;

            if (row < M && aCol < K) {
                tileA[writeIdx][threadIdx.y][threadIdx.x] = A[row * K + aCol];
            } else {
                tileA[writeIdx][threadIdx.y][threadIdx.x] = 0.0f;
            }

            if (bRow < K && col < N) {
                tileB[writeIdx][threadIdx.y][threadIdx.x] = B[bRow * N + col];
            } else {
                tileB[writeIdx][threadIdx.y][threadIdx.x] = 0.0f;
            }
        }

        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            acc += tileA[readIdx][threadIdx.y][k] * tileB[readIdx][k][threadIdx.x];
        }

        __syncthreads();  
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}
