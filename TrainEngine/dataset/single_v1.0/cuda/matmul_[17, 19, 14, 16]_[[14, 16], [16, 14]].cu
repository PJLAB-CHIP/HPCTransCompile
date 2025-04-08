extern "C" __global__ void __launch_bounds__(7) default_function_kernel(float* __restrict__ T_matmul, float* __restrict__ left_matrix, float* __restrict__ right_matrix) {
  float T_matmul_local[4];
  __shared__ float left_matrix_shared[16];
  __shared__ float right_matrix_shared[112];
  for (int ax1_c_outer_inner_init = 0; ax1_c_outer_inner_init < 2; ++ax1_c_outer_inner_init) {
    for (int ax0_c_inner_init = 0; ax0_c_inner_init < 2; ++ax0_c_inner_init) {
      T_matmul_local[((ax0_c_inner_init * 2) + ax1_c_outer_inner_init)] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 2; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 3; ++ax0_ax1_fused_outer_outer) {
      if (((ax0_ax1_fused_outer_outer * 7) + ((int)threadIdx.x)) < 16) {
        left_matrix_shared[((ax0_ax1_fused_outer_outer * 7) + ((int)threadIdx.x))] = left_matrix[((((((int)blockIdx.x) * 32) + ((((ax0_ax1_fused_outer_outer * 7) + ((int)threadIdx.x)) >> 3) * 16)) + (k_outer_outer * 8)) + (((ax0_ax1_fused_outer_outer * 7) + ((int)threadIdx.x)) & 7))];
      }
    }
    for (int ax0_ax1_fused_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_1 < 16; ++ax0_ax1_fused_outer_outer_1) {
      right_matrix_shared[((ax0_ax1_fused_outer_outer_1 * 7) + ((int)threadIdx.x))] = right_matrix[(((k_outer_outer * 112) + (ax0_ax1_fused_outer_outer_1 * 7)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      for (int ax1_c_outer_inner = 0; ax1_c_outer_inner < 2; ++ax1_c_outer_inner) {
        for (int ax0_c_inner = 0; ax0_c_inner < 2; ++ax0_c_inner) {
          T_matmul_local[((ax0_c_inner * 2) + ax1_c_outer_inner)] = (T_matmul_local[((ax0_c_inner * 2) + ax1_c_outer_inner)] + (left_matrix_shared[((ax0_c_inner * 8) + k_outer_inner)] * right_matrix_shared[(((k_outer_inner * 14) + (((int)threadIdx.x) * 2)) + ax1_c_outer_inner)]));
        }
      }
    }
  }
  for (int ax0_inner = 0; ax0_inner < 2; ++ax0_inner) {
    for (int ax1_inner = 0; ax1_inner < 2; ++ax1_inner) {
      T_matmul[((((((int)blockIdx.x) * 28) + (ax0_inner * 14)) + (((int)threadIdx.x) * 2)) + ax1_inner)] = T_matmul_local[((ax0_inner * 2) + ax1_inner)];
    }
  }
}

