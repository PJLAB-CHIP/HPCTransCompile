
    Task: Translate the given CUDA code to its equivalent high-performance CPU C code.
    Context: You are provided with a CUDA code snippet that needs to be translated into CPU C code. The translation should preserve the same functionality as much as possible. Focus on translating the CUDA-specific parallel constructs into constructs supported by the CPU, such as using OpenMP for parallelism. The resulting CPU C code should be complete and ready to compile.

    Example 1:
    Input CUDA Code: extern "C" __global__ void __launch_bounds__(18) default_function_kernel(float* __restrict__ compute, float* __restrict__ data) {
  compute[((((int)blockIdx.x) * 18) + ((int)threadIdx.x))] = atanf(data[((((int)blockIdx.x) * 18) + ((int)threadIdx.x))]);
}


    Input Tensor Shape: [[17, 12, 7, 9]]
    Output C Code: void default_function_kernel(float* compute, float* data) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 1428; ++i0_i1_fused_i2_fused) {
    for (int32_t i3_s = 0; i3_s < 9; ++i3_s) {
      compute[((i0_i1_fused_i2_fused * 9) + i3_s)] = atanf(data[((i0_i1_fused_i2_fused * 9) + i3_s)]);
    }
  }
}


    //|End-of-Code|

    Example 2:
    Input CUDA Code: extern "C" __global__ void __launch_bounds__(2) default_function_kernel(float* __restrict__ T_matmul, float* __restrict__ left_matrix, float* __restrict__ right_matrix) {
  float T_matmul_local[2];
  __shared__ float left_matrix_shared[4];
  __shared__ float right_matrix_shared[4];
  T_matmul_local[0] = 0.000000e+00f;
  T_matmul_local[1] = 0.000000e+00f;
  for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 2; ++ax0_ax1_fused_outer_outer) {
    left_matrix_shared[((ax0_ax1_fused_outer_outer * 2) + ((int)threadIdx.x))] = left_matrix[((ax0_ax1_fused_outer_outer * 2) + ((int)threadIdx.x))];
  }
  for (int ax0_ax1_fused_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_1 < 2; ++ax0_ax1_fused_outer_outer_1) {
    right_matrix_shared[((ax0_ax1_fused_outer_outer_1 * 2) + ((int)threadIdx.x))] = right_matrix[((ax0_ax1_fused_outer_outer_1 * 2) + ((int)threadIdx.x))];
  }
  __syncthreads();
  for (int k_inner = 0; k_inner < 2; ++k_inner) {
    T_matmul_local[0] = (T_matmul_local[0] + (left_matrix_shared[((((int)threadIdx.x) * 2) + k_inner)] * right_matrix_shared[(k_inner * 2)]));
    T_matmul_local[1] = (T_matmul_local[1] + (left_matrix_shared[((((int)threadIdx.x) * 2) + k_inner)] * right_matrix_shared[((k_inner * 2) + 1)]));
  }
  T_matmul[(((int)threadIdx.x) * 2)] = T_matmul_local[0];
  T_matmul[((((int)threadIdx.x) * 2) + 1)] = T_matmul_local[1];
}


    Input Tensor Shape: [[2, 2], [2, 2]]
    Output C Code: void default_function_kernel(float* T_matmul, float* left_matrix, float* right_matrix) {
  for (int32_t ax1_outer_outer_outer = 0; ax1_outer_outer_outer < 2; ++ax1_outer_outer_outer) {
    for (int32_t ax0_inner_init = 0; ax0_inner_init < 2; ++ax0_inner_init) {
      T_matmul[((ax0_inner_init * 2) + ax1_outer_outer_outer)] = 0.000000e+00f;
    }
    for (int32_t k_inner = 0; k_inner < 2; ++k_inner) {
      for (int32_t ax0_inner = 0; ax0_inner < 2; ++ax0_inner) {
        T_matmul[((ax0_inner * 2) + ax1_outer_outer_outer)] = (T_matmul[((ax0_inner * 2) + ax1_outer_outer_outer)] + (left_matrix[((ax0_inner * 2) + k_inner)] * right_matrix[((k_inner * 2) + ax1_outer_outer_outer)]));
      }
    }
  }
}


    //|End-of-Code|

    Now translate the following CUDA code to its equivalent high-performance CPU C code:
    Input CUDA Code: extern "C" __global__ void __launch_bounds__(34) default_function_kernel(float* __restrict__ compute, float* __restrict__ data) {
  compute[((((int)blockIdx.x) * 34) + ((int)threadIdx.x))] = sinhf(data[((((int)blockIdx.x) * 34) + ((int)threadIdx.x))]);
}


    Input Tensor Shape: [[13, 7, 14, 17]]
    Print only a single C function implementation, ending with the comment '|End-of-Code|'.
    