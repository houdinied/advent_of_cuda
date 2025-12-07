#include <iostream>
#include <chrono>
#include <cstring>
#include <iomanip>

void transpose_cpu_naive(const float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

__global__ void transpose_gpu_shared(const float* input, float* output, int rows, int cols) {
  __shared__ float tile[32][33]; // 16.7M bank conflicts to just 108k.

  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;

  if (x < cols && y < rows) {
      tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
  }
  __syncthreads(); // Wait for all threads to finish loading

  int x_out = blockIdx.y * 32 + threadIdx.x;
  int y_out = blockIdx.x * 32 + threadIdx.y;

  if (x_out < rows && y_out < cols) {
      output[y_out * rows + x_out] = tile[threadIdx.x][threadIdx.y];
  }
}

__global__ void register_tiled(const float* input, float* output, int rows, int cols) {
  // Each thread handles a 4x4 sub-tile
  constexpr int TILE_DIM = 32;
  constexpr int BLOCK_ROWS = 8;  // 32/4 = 8 threads in y-dimension

  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  float reg[4];

  int x      = blockIdx.x * TILE_DIM + threadIdx.x;
  int y_base = blockIdx.y * TILE_DIM + threadIdx.y;

  // Load 4 elements per thread into registers (coalesced reads)
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    int y = y_base + i * BLOCK_ROWS;
    if (x < cols && y < rows) {
      reg[i] = input[y * cols + x];
    }
  }

  // Write from registers to shared memory
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    tile[threadIdx.y + i * BLOCK_ROWS][threadIdx.x] = reg[i];
  }

  __syncthreads();

  // Read from shared memory in transposed pattern into registers
  x      = blockIdx.y * TILE_DIM + threadIdx.x;
  y_base = blockIdx.x * TILE_DIM + threadIdx.y;

  #pragma unroll
  for (int i = 0; i < 4; i++) {
    reg[i] = tile[threadIdx.x][threadIdx.y + i * BLOCK_ROWS];
  }

  // Write from registers to output (coalesced writes)
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    int y = y_base + i * BLOCK_ROWS;
    if (x < rows && y < cols) {
      output[y * rows + x] = reg[i];
    }
  }
}


double benchmark_transpose(void (*transpose_fn)(const float*, float*, int, int),
                          const float* input, float* output, int rows, int cols,
                          int warmup_iters, int bench_iters) {
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        transpose_fn(input, output, rows, cols);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; i++) {
        transpose_fn(input, output, rows, cols);
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    return diff.count() / bench_iters;
}

int main() {
    const int rows = 4096;
    const int cols = 4096;
    const int warmup_iters = 5;
    const int bench_iters = 20;

    size_t matrix_size = rows * cols * sizeof(float);
    
    float *input, *output, *cpu_output;
    cudaMallocManaged(&input, matrix_size);
    cudaMallocManaged(&output, matrix_size);
    cpu_output = new float[rows * cols];

    for (int i = 0; i < rows * cols; i++) {
        input[i] = static_cast<float>(i);
    }

    // CPU benchmark
    double time_naive = benchmark_transpose(transpose_cpu_naive, input, cpu_output, rows, cols, warmup_iters, bench_iters);
    double bandwidth_naive = (2.0 * matrix_size / time_naive) / 1e9;

    std::cout << "Naive transpose:\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(6) << time_naive * 1000 << " ms\n";
    std::cout << "  Bandwidth: " << std::setprecision(2) << bandwidth_naive << " GB/s\n\n";

    // GPU benchmark
    dim3 blockDim(32, 32);
    dim3 gridDim((cols + 31) / 32, (rows + 31) / 32);
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        transpose_gpu_shared<<<gridDim, blockDim>>>(input, output, rows, cols);
    }
    cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; i++) {
        transpose_gpu_shared<<<gridDim, blockDim>>>(input, output, rows, cols);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    double time_blocked = diff.count() / bench_iters;
    double bandwidth_blocked = (2.0 * matrix_size / time_blocked) / 1e9;

    std::cout << "CUDA Blocked transpose (32x32):\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(6) << time_blocked * 1000 << " ms\n";
    std::cout << "  Bandwidth: " << std::setprecision(2) << bandwidth_blocked << " GB/s\n\n";


    dim3 blockDim_reg(32, 8);  // 32x8 threads, each handles 4 elements
    dim3 gridDim_reg((cols + 31) / 32, (rows + 31) / 32);

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        register_tiled<<<gridDim_reg, blockDim_reg>>>(input, output, rows, cols);
    }
    cudaDeviceSynchronize();

    auto start_reg = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; i++) {
        register_tiled<<<gridDim_reg, blockDim_reg>>>(input, output, rows, cols);
    }
    cudaDeviceSynchronize();
    auto end_reg = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff_reg = end_reg - start_reg;
    double time_reg = diff_reg.count() / bench_iters;
    double bandwidth_reg = (2.0 * matrix_size / time_reg) / 1e9;

    std::cout << "CUDA Register Tiled transpose (4 elem/thread):\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(6) << time_reg * 1000 << " ms\n";
    std::cout << "  Bandwidth: " << std::setprecision(2) << bandwidth_reg << " GB/s\n\n";

    bool correct = true;
    for (int i = 0; i < rows * cols && correct; i++) {
        if (output[i] != cpu_output[i]) {
            correct = false;
            std::cout << "Mismatch at index " << i << "\n";
        }
    }
    std::cout << "Verification: " << (correct ? "PASSED" : "FAILED") << "\n";

    cudaFree(input);
    cudaFree(output);
    delete[] cpu_output;
    return 0;
}
