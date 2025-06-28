extern "C" __global__ void __launch_bounds__(4) default_function_kernel(float* __restrict__ T_matmul, float* __restrict__ left_matrix, float* __restrict__ right_matrix) {
  float T_matmul_local[8];
  __shared__ float left_matrix_shared[68];
  __shared__ float right_matrix_shared[136];
  for (int ax1_c_outer_inner_init = 0; ax1_c_outer_inner_init < 2; ++ax1_c_outer_inner_init) {
    for (int ax0_c_inner_init = 0; ax0_c_inner_init < 2; ++ax0_c_inner_init) {
      for (int ax1_c_inner_init = 0; ax1_c_inner_init < 2; ++ax1_c_inner_init) {
        T_matmul_local[(((ax0_c_inner_init * 4) + (ax1_c_outer_inner_init * 2)) + ax1_c_inner_init)] = 0.000000e+00f;
      }
    }
  }
  for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 17; ++ax0_ax1_fused_outer_outer) {
    left_matrix_shared[((ax0_ax1_fused_outer_outer * 4) + ((int)threadIdx.x))] = left_matrix[(((((int)blockIdx.x) * 68) + (ax0_ax1_fused_outer_outer * 4)) + ((int)threadIdx.x))];
  }
  for (int ax0_ax1_fused_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_1 < 17; ++ax0_ax1_fused_outer_outer_1) {
    *(float2*)(right_matrix_shared + ((ax0_ax1_fused_outer_outer_1 * 8) + (((int)threadIdx.x) * 2))) = *(float2*)(right_matrix + ((ax0_ax1_fused_outer_outer_1 * 8) + (((int)threadIdx.x) * 2)));
  }
  __syncthreads();
  for (int ax1_c_outer_inner = 0; ax1_c_outer_inner < 2; ++ax1_c_outer_inner) {
    for (int k_inner = 0; k_inner < 17; ++k_inner) {
      for (int ax0_c_inner = 0; ax0_c_inner < 2; ++ax0_c_inner) {
        for (int ax1_c_inner = 0; ax1_c_inner < 2; ++ax1_c_inner) {
          T_matmul_local[(((ax0_c_inner * 4) + (ax1_c_outer_inner * 2)) + ax1_c_inner)] = (T_matmul_local[(((ax0_c_inner * 4) + (ax1_c_outer_inner * 2)) + ax1_c_inner)] + (left_matrix_shared[((((((int)threadIdx.x) >> 1) * 34) + (ax0_c_inner * 17)) + k_inner)] * right_matrix_shared[((((k_inner * 8) + ((((int)threadIdx.x) & 1) * 4)) + (ax1_c_outer_inner * 2)) + ax1_c_inner)]));
        }
      }
    }
  }
  for (int ax0_inner = 0; ax0_inner < 2; ++ax0_inner) {
    for (int ax1_inner = 0; ax1_inner < 4; ++ax1_inner) {
      T_matmul[(((((((int)blockIdx.x) * 32) + ((((int)threadIdx.x) >> 1) * 16)) + (ax0_inner * 8)) + ((((int)threadIdx.x) & 1) * 4)) + ax1_inner)] = T_matmul_local[((ax0_inner * 4) + ax1_inner)];
    }
  }
}

