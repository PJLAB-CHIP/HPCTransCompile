extern "C" __global__ void __launch_bounds__(2) default_function_kernel(float* __restrict__ T_matmul, float* __restrict__ left_matrix, float* __restrict__ right_matrix) {
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

