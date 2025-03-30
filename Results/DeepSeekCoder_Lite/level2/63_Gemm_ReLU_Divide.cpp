#include <torch/extension.h>
#include <omp.h>
#include <iostream>

#define TILE 16

template <typename scalar_t>
void unrolled_tiled_gemm_kernel_cpu(
    const scalar_t* x,       // [M, K]
    const scalar_t* weight,  // [N, K]
    const scalar_t* bias,    // [N]
    scalar_t* output,        // [M, N]
    float divisor,
    int M,  // number of rows in x
    int K,  // number of columns in x (in_features)
    int N   // number of rows in weight (out_features)
) {
    int rowBase = omp_get_thread_num() * TILE;
    int colBase = 0;  // Assuming we process tiles sequentially for simplicity
    int localRow = 0;
    int localCol = 0;
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

        __asm__("" : : "r"(sum));  // Ensure sum is not optimized away
    }

    // Write output with bias addition, ReLU activation, and division
    if (globalRow < M && globalCol < N) {
        sum += bias[globalCol];
        scalar_t result = (sum > 0) ? (sum / divisor) : static_cast<scalar_t>(0);
        output[globalRow * N + globalCol] = result;
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

    #pragma omp parallel for
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            unrolled_tiled_gemm_kernel_cpu<typename decltype(x)::value_type>(
                x.data_ptr<typename decltype(x)::value_type>(),
                weight.data_ptr<typename decltype(weight)::value_type>(),
                bias.data_ptr<typename decltype(bias)::value_type>(),
                output.data_ptr<typename decltype(output)::value_type>(),
                divisor,
                M, K, N
            );
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &linear_relu_div_cpu_forward, "Unrolled Tiled GEMM with ReLU and Div (CPU)");
}