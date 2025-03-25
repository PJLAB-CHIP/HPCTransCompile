#include <torch/extension.h>
#include <omp.h>

#define TILE 16

// This function performs tiled GEMM with manual unrolling of the inner loop to reduce loop overhead.
// It computes: output[m, n] = (ReLU(dot(x[m, :], weight[n, :]) + bias[n])) / divisor
// where x is [M, K] and weight is [N, K] (each row corresponds to an output neuron).

template <typename scalar_t>
void unrolled_tiled_gemm_cpu(
    const scalar_t* x,       // [M, K]
    const scalar_t* weight,    // [N, K]
    const scalar_t* bias,      // [N]
    scalar_t* output,          // [M, N]
    const float divisor,
    const int M,  // number of rows in x
    const int K,  // number of columns in x (in_features)
    const int N   // number of rows in weight (out_features)
) {
    #pragma omp parallel for collapse(2)
    for (int rowBase = 0; rowBase < M; rowBase += TILE) {
        for (int colBase = 0; colBase < N; colBase += TILE) {
            for (int localRow = 0; localRow < TILE; ++localRow) {
                for (int localCol = 0; localCol < TILE; ++localCol) {
                    int globalRow = rowBase + localRow;
                    int globalCol = colBase + localCol;

                    scalar_t sum = 0;

                    // Loop over tiles in the K dimension
                    int numTiles = (K + TILE - 1) / TILE;
                    for (int t = 0; t < numTiles; t++) {
                        int tileStart = t * TILE;
                        int aCol = tileStart + localCol;
                        if (globalRow < M && aCol < K)
                            sum += x[globalRow * K + aCol] * weight[globalCol * K + aCol];
                    }

                    // Write output with bias addition, ReLU activation, and division
                    if (globalRow < M && globalCol < N) {
                        sum += bias[globalCol];
                        scalar_t result = (sum > 0) ? (sum / divisor) : static_cast<scalar_t>(0);
                        output[globalRow * N + globalCol] = result;
                    }
                }
            }
        }
    }
}

// CPU forward function interfacing with PyTorch

torch::Tensor linear_relu_div_cpu_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor
) {
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = weight.size(0);

    auto output = torch::empty({M, N}, x.options());

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "unrolled_tiled_gemm_cpu", ([&] {
        unrolled_tiled_gemm_cpu<scalar_t>(
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
    m.def("forward", &linear_relu_div_cpu_forward, "Unrolled Tiled GEMM with ReLU and Div (CPU)");
}