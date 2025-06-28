
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
    Input CUDA Code: extern "C" __global__ void default_function_kernel(float* __restrict__ A, float* __restrict__ W, float* __restrict__ group_conv2d_nchw) {
  float group_conv2d_nchw_local[1];
  __shared__ float pad_temp_shared[1];
  __shared__ float W_shared[1];
  group_conv2d_nchw_local[0] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
    for (int ry_outer = 0; ry_outer < 3; ++ry_outer) {
      for (int rx_outer = 0; rx_outer < 3; ++rx_outer) {
        __syncthreads();
        pad_temp_shared[0] = A[(((((((((int)blockIdx.z) * 2097152) + ((((int)blockIdx.y) >> 4) * 524288)) + (rc_outer * 65536)) + ((((int)blockIdx.x) / 254) * 256)) + (ry_outer * 256)) + rx_outer) + (((int)blockIdx.x) % 254))];
        W_shared[0] = W[((((((int)blockIdx.y) * 72) + (rc_outer * 9)) + (ry_outer * 3)) + rx_outer)];
        __syncthreads();
        group_conv2d_nchw_local[0] = (group_conv2d_nchw_local[0] + (pad_temp_shared[0] * W_shared[0]));
      }
    }
  }
  group_conv2d_nchw[(((((int)blockIdx.z) * 4129024) + (((int)blockIdx.y) * 64516)) + ((int)blockIdx.x))] = group_conv2d_nchw_local[0];
}


    Input Tensor Shape: [[4, 32, 256, 256], [64, 8, 3, 3]]
    Print only a single C function implementation, ending with the comment '|End-of-Code|'.
    