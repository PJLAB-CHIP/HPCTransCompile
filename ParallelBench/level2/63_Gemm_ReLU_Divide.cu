#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE 16

// This kernel performs tiled GEMM with manual unrolling of the inner loop to reduce loop overhead.
// It computes: output[m, n] = (ReLU(dot(x[m, :], weight[n, :]) + bias[n])) / divisor
// where x is [M, K] and weight is [N, K] (each row corresponds to an output neuron).

template <typename scalar_t>
__global__ void unrolled_tiled_gemm_kernel(
    const scalar_t* __restrict__ x,       // [M, K]
    const scalar_t* __restrict__ weight,    // [N, K]
    const scalar_t* __restrict__ bias,      // [N]
    scalar_t* __restrict__ output,          // [M, N]
    const float divisor,
    const int M,  // number of rows in x
    const int K,  // number of columns in x (in_features)
    const int N   // number of rows in weight (out_features)
) {
    int rowBase = blockIdx.y * TILE;
    int colBase = blockIdx.x * TILE;
    int localRow = threadIdx.y;
    int localCol = threadIdx.x;
    int globalRow = rowBase + localRow;
    int globalCol = colBase + localCol;

    scalar_t sum = 0;

    __shared__ scalar_t sA[TILE][TILE];
    __shared__ scalar_t sB[TILE][TILE];

    // Loop over tiles in the K dimension
    int numTiles = (K + TILE - 1) / TILE;
    #pragma unroll
    for (int t = 0; t < numTiles; t++) {
        int tileStart = t * TILE;
        int aCol = tileStart + localCol;
        if (globalRow < M && aCol < K)
            sA[localRow][localCol] = x[globalRow * K + aCol];
        else
            sA[localRow][localCol] = static_cast<scalar_t>(0);

        int weightRow = colBase + localRow;  // corresponds to output neuron index
        int weightCol = tileStart + localCol;
        // Load weight in transposed fashion for coalesced access
        if (weightRow < N && weightCol < K)
            sB[localCol][localRow] = weight[weightRow * K + weightCol];
        else
            sB[localCol][localRow] = static_cast<scalar_t>(0);

        __syncthreads();

        // Manually unroll the inner product computation for the tile
        // TILE is 16, so we unroll 16 iterations explicitly
        sum += sA[localRow][0] * sB[0][localCol];
        sum += sA[localRow][1] * sB[1][localCol];
        sum += sA[localRow][2] * sB[2][localCol];
        sum += sA[localRow][3] * sB[3][localCol];
        sum += sA[localRow][4] * sB[4][localCol];
        sum += sA[localRow][5] * sB[5][localCol];
        sum += sA[localRow][6] * sB[6][localCol];
        sum += sA[localRow][7] * sB[7][localCol];
        sum += sA[localRow][8] * sB[8][localCol];
        sum += sA[localRow][9] * sB[9][localCol];
        sum += sA[localRow][10] * sB[10][localCol];
        sum += sA[localRow][11] * sB[11][localCol];
        sum += sA[localRow][12] * sB[12][localCol];
        sum += sA[localRow][13] * sB[13][localCol];
        sum += sA[localRow][14] * sB[14][localCol];
        sum += sA[localRow][15] * sB[15][localCol];

        __syncthreads();
    }

    // Write output with bias addition, ReLU activation, and division
    if (globalRow < M && globalCol < N) {
        sum += bias[globalCol];
        scalar_t result = (sum > 0) ? (sum / divisor) : static_cast<scalar_t>(0);
        output[globalRow * N + globalCol] = result;
    }
}

// CUDA forward function interfacing with PyTorch

torch::Tensor linear_relu_div_cuda_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor
) {
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = weight.size(0);

    auto output = torch::empty({M, N}, x.options());

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "unrolled_tiled_gemm_cuda", ([&] {
        unrolled_tiled_gemm_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            divisor,
            M, K, N
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &linear_relu_div_cuda_forward, "Unrolled Tiled GEMM with ReLU and Div (CUDA)");
}
