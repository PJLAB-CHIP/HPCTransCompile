extern "C" __global__ void __launch_bounds__(2) default_function_kernel(float* __restrict__ T_matmul, float* __restrict__ left_matrix, float* __restrict__ right_matrix) {
  float T_matmul_local[16];
  __shared__ float left_matrix_shared[72];
  __shared__ float right_matrix_shared[36];
  for (int ax0_c_outer_inner_init = 0; ax0_c_outer_inner_init < 2; ++ax0_c_outer_inner_init) {
    for (int ax1_c_outer_inner_init = 0; ax1_c_outer_inner_init < 2; ++ax1_c_outer_inner_init) {
      for (int ax1_c_inner_init = 0; ax1_c_inner_init < 2; ++ax1_c_inner_init) {
        T_matmul_local[(((ax0_c_outer_inner_init * 4) + (ax1_c_outer_inner_init * 2)) + ax1_c_inner_init)] = 0.000000e+00f;
        T_matmul_local[((((ax0_c_outer_inner_init * 4) + (ax1_c_outer_inner_init * 2)) + ax1_c_inner_init) + 8)] = 0.000000e+00f;
      }
    }
  }
  for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 12; ++ax0_ax1_fused_outer_outer) {
    *(float3*)(left_matrix_shared + ((ax0_ax1_fused_outer_outer * 6) + (((int)threadIdx.x) * 3))) = *(float3*)(left_matrix + ((ax0_ax1_fused_outer_outer * 6) + (((int)threadIdx.x) * 3)));
  }
  for (int ax0_ax1_fused_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_1 < 18; ++ax0_ax1_fused_outer_outer_1) {
    right_matrix_shared[((ax0_ax1_fused_outer_outer_1 * 2) + ((int)threadIdx.x))] = right_matrix[(((((ax0_ax1_fused_outer_outer_1 >> 1) * 8) + (((int)blockIdx.x) * 4)) + ((ax0_ax1_fused_outer_outer_1 & 1) * 2)) + ((int)threadIdx.x))];
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 9; ++k_outer_inner) {
    for (int ax0_c_outer_inner = 0; ax0_c_outer_inner < 2; ++ax0_c_outer_inner) {
      for (int ax1_c_outer_inner = 0; ax1_c_outer_inner < 2; ++ax1_c_outer_inner) {
        for (int ax1_c_inner = 0; ax1_c_inner < 2; ++ax1_c_inner) {
          T_matmul_local[(((ax0_c_outer_inner * 4) + (ax1_c_outer_inner * 2)) + ax1_c_inner)] = (T_matmul_local[(((ax0_c_outer_inner * 4) + (ax1_c_outer_inner * 2)) + ax1_c_inner)] + (left_matrix_shared[(((((int)threadIdx.x) * 18) + (ax0_c_outer_inner * 9)) + k_outer_inner)] * right_matrix_shared[(((k_outer_inner * 4) + (ax1_c_outer_inner * 2)) + ax1_c_inner)]));
          T_matmul_local[((((ax0_c_outer_inner * 4) + (ax1_c_outer_inner * 2)) + ax1_c_inner) + 8)] = (T_matmul_local[((((ax0_c_outer_inner * 4) + (ax1_c_outer_inner * 2)) + ax1_c_inner) + 8)] + (left_matrix_shared[((((((int)threadIdx.x) * 18) + (ax0_c_outer_inner * 9)) + k_outer_inner) + 36)] * right_matrix_shared[(((k_outer_inner * 4) + (ax1_c_outer_inner * 2)) + ax1_c_inner)]));
        }
      }
    }
  }
  for (int ax0_inner = 0; ax0_inner < 2; ++ax0_inner) {
    for (int ax1_inner = 0; ax1_inner < 4; ++ax1_inner) {
      T_matmul[((((((int)threadIdx.x) * 16) + (ax0_inner * 8)) + (((int)blockIdx.x) * 4)) + ax1_inner)] = T_matmul_local[((ax0_inner * 4) + ax1_inner)];
      T_matmul[(((((((int)threadIdx.x) * 16) + (ax0_inner * 8)) + (((int)blockIdx.x) * 4)) + ax1_inner) + 32)] = T_matmul_local[(((ax0_inner * 4) + ax1_inner) + 8)];
    }
  }
}

