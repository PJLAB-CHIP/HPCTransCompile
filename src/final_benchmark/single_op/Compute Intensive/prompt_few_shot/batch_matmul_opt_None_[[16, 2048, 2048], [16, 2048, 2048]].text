
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
    Input CUDA Code: extern "C" __global__ void __launch_bounds__(64) default_function_kernel(float* __restrict__ T_batch_matmul_NT, float* __restrict__ x, float* __restrict__ y) {
  float T_batch_matmul_NT_local[64];
  __shared__ float x_shared[512];
  __shared__ float y_shared[512];
  float x_shared_local[8];
  float y_shared_local[8];
  for (int i_c_init = 0; i_c_init < 8; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 8; ++j_c_init) {
      T_batch_matmul_NT_local[((i_c_init * 8) + j_c_init)] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 256; ++k_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
      x_shared[(((((int)threadIdx.y) * 64) + (ax1_inner * 8)) + ((int)threadIdx.x))] = x[((((((((int)blockIdx.z) * 4194304) + (((int)blockIdx.y) * 131072)) + (((int)threadIdx.y) * 16384)) + (ax1_inner * 2048)) + (k_outer * 8)) + ((int)threadIdx.x))];
    }
    #pragma unroll
    for (int ax1_inner_1 = 0; ax1_inner_1 < 8; ++ax1_inner_1) {
      y_shared[(((((int)threadIdx.y) * 64) + (ax1_inner_1 * 8)) + ((int)threadIdx.x))] = y[((((((((int)blockIdx.z) * 4194304) + (((int)blockIdx.x) * 131072)) + (((int)threadIdx.y) * 16384)) + (ax1_inner_1 * 2048)) + (k_outer * 8)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
      #pragma unroll
      for (int ax1 = 0; ax1 < 8; ++ax1) {
        x_shared_local[ax1] = x_shared[(((((int)threadIdx.y) * 64) + (ax1 * 8)) + k_inner)];
      }
      #pragma unroll
      for (int ax1_1 = 0; ax1_1 < 8; ++ax1_1) {
        y_shared_local[ax1_1] = y_shared[(((((int)threadIdx.x) * 64) + (ax1_1 * 8)) + k_inner)];
      }
      for (int i_c = 0; i_c < 8; ++i_c) {
        #pragma unroll
        for (int j_c = 0; j_c < 8; ++j_c) {
          T_batch_matmul_NT_local[((i_c * 8) + j_c)] = (T_batch_matmul_NT_local[((i_c * 8) + j_c)] + (x_shared_local[i_c] * y_shared_local[j_c]));
        }
      }
    }
  }
  for (int i_inner_inner = 0; i_inner_inner < 8; ++i_inner_inner) {
    #pragma unroll
    for (int j_inner_inner = 0; j_inner_inner < 8; ++j_inner_inner) {
      T_batch_matmul_NT[(((((((((int)blockIdx.z) * 4194304) + (((int)blockIdx.y) * 131072)) + (((int)threadIdx.y) * 16384)) + (i_inner_inner * 2048)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + j_inner_inner)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + j_inner_inner)];
    }
  }
}


    Input Tensor Shape: [[16, 2048, 2048], [16, 2048, 2048]]
    Print only a single C function implementation, ending with the comment '|End-of-Code|'.
    