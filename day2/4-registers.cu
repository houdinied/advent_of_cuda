__global__ void registers(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    __shared__ float tileA[2][32][33];
    __shared__ float tileB[2][32][33];

    // Each thread computes an 8×8 patch of C
    // With 4×4 threads per block, the block computes a 32×32 tile
    int baseRow = blockIdx.y * 32 + threadIdx.y * 8;
    int baseCol = blockIdx.x * 32 + threadIdx.x * 8;

    float acc[8][8];
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    int writeIdx = 0;
    int readIdx = 1;

    // Thread ID within block (0-15)
    int tid = threadIdx.y * 4 + threadIdx.x;

    int numTiles = (K + 31) / 32;

    // Cooperative loading: 16 threads load 1024 elements (32×32)
    // Each thread loads 64 elements in an 8×8 block
    // Thread tid loads elements at positions [tid/4 * 8 : tid/4 * 8 + 7] × [tid%4 * 8 : tid%4 * 8 + 7]

    auto loadTile = [&](int bufIdx, int tileIdx) {
        int loadRowBase = (tid / 4) * 8;  // 0, 8, 16, 24
        int loadColBase = (tid % 4) * 8;  // 0, 8, 16, 24

        // Load 8×8 block from A
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                int row = blockIdx.y * 32 + loadRowBase + i;
                int col = tileIdx * 32 + loadColBase + j;

                if (row < M && col < K) {
                    tileA[bufIdx][loadRowBase + i][loadColBase + j] = A[row * K + col];
                } else {
                    tileA[bufIdx][loadRowBase + i][loadColBase + j] = 0.0f;
                }
            }
        }

        // Load 8×8 block from B
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                int row = tileIdx * 32 + loadRowBase + i;
                int col = blockIdx.x * 32 + loadColBase + j;

                if (row < K && col < N) {
                    tileB[bufIdx][loadRowBase + i][loadColBase + j] = B[row * N + col];
                } else {
                    tileB[bufIdx][loadRowBase + i][loadColBase + j] = 0.0f;
                }
            }
        }
    };

    // Load first tile
    loadTile(0, 0);
    __syncthreads();

    // Main computation loop
    for (int tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
        // Swap buffers
        readIdx = writeIdx;
        writeIdx = 1 - writeIdx;

        if (tileIdx + 1 < numTiles) {
            loadTile(writeIdx, tileIdx + 1);
        }

        // Compute using current tile
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            // Load 8 elements from tileA for this thread's 8 rows
            float a[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                a[i] = tileA[readIdx][threadIdx.y * 8 + i][k];
            }

            // Load 8 elements from tileB for this thread's 8 columns
            float b[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                b[j] = tileB[readIdx][k][threadIdx.x * 8 + j];
            }

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            int row = baseRow + i;
            int col = baseCol + j;
            if (row < M && col < N) {
                C[row * N + col] = alpha * acc[i][j] + beta * C[row * N + col];
            }
        }
    }
}
