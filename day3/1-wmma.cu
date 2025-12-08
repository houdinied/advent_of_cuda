//#include <iostream>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

__global__
void wmma_gemm(int M, int N, int K, float alpha, const __half *A, const __half *B, float beta, float *C) {
    // Size: 32x16 to hold 2x1 warp rows × 16 K-elements
    __shared__ __half tileA[2][32][16];  // [buffer][rows][cols]
    __shared__ __half tileB[2][16][32];  // [buffer][rows][cols]

    int writeIdx = 0;
    int readIdx = 1;

    // With 128 threads = 4 warps per block
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;

    int warpRow = warpId / 2;  // 0 or 1
    int warpCol = warpId % 2;  // 0 or 1

    // Global tile position
    int warpM = blockIdx.y * 2 + warpRow;
    int warpN = blockIdx.x * 2 + warpCol;

    // Bounds check
    if (warpM * 16 >= M || warpN * 16 >= N) return;

    int tid = threadIdx.x;

    // For loading 32x16 tile of A: 512 elements / 128 threads = 4 elements/thread
    // For loading 16x32 tile of B: 512 elements / 128 threads = 4 elements/thread
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);
    {
        int k = 0;

        // Load A tile: block covers rows [blockIdx.y*32 : blockIdx.y*32+32]
        // Load 32x16 chunk from global A
        for (int i = tid; i < 32 * 16; i += 128) {
            int row = i / 16;
            int col = i % 16;
            int globalRow = blockIdx.y * 32 + row;
            int globalCol = k + col;

            if (globalRow < M && globalCol < K) {
                tileA[0][row][col] = A[globalRow * K + globalCol];
            } else {
                tileA[0][row][col] = __float2half(0.0f);
            }
        }

        // Load B tile: block covers cols [blockIdx.x*32 : blockIdx.x*32+32]
        // Load 16x32 chunk from global B
        for (int i = tid; i < 16 * 32; i += 128) {
            int row = i / 32;
            int col = i % 32;
            int globalRow = k + row;
            int globalCol = blockIdx.x * 32 + col;

            if (globalRow < K && globalCol < N) {
                tileB[0][row][col] = B[globalRow * N + globalCol];
            } else {
                tileB[0][row][col] = __float2half(0.0f);
            }
        }
    }
    __syncthreads();

    for (int k = 0; k < K; k += 16) {
        // Swap buffers
        readIdx = writeIdx;
        writeIdx = 1 - writeIdx;

        // Load NEXT tile into writeIdx (if not last iteration)
        if (k + 16 < K) {
            int nextK = k + 16;

            for (int i = tid; i < 32 * 16; i += 128) {
                int row = i / 16;
                int col = i % 16;
                int globalRow = blockIdx.y * 32 + row;
                int globalCol = nextK + col;

                if (globalRow < M && globalCol < K) {
                    tileA[writeIdx][row][col] = A[globalRow * K + globalCol];
                } else {
                    tileA[writeIdx][row][col] = __float2half(0.0f);
                }
            }

            for (int i = tid; i < 16 * 32; i += 128) {
                int row = i / 32;
                int col = i % 32;
                int globalRow = nextK + row;
                int globalCol = blockIdx.x * 32 + col;

                if (globalRow < K && globalCol < N) {
                    tileB[writeIdx][row][col] = B[globalRow * N + globalCol];
                } else {
                    tileB[writeIdx][row][col] = __float2half(0.0f);
                }
            }
        }

        wmma::load_matrix_sync(a_frag, &tileA[readIdx][warpRow * 16][0], 16);
        wmma::load_matrix_sync(b_frag, &tileB[readIdx][0][warpCol * 16], 32);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < c_frag.num_elements; i++) {
        c_frag.x[i] *= alpha;
    }

    int cRow = warpM * 16;
    int cCol = warpN * 16;
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}
