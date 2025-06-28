extern "C" __global__ void __launch_bounds__(4) default_function_kernel(float* __restrict__ T_matmul, float* __restrict__ left_matrix, float* __restrict__ right_matrix) {
  float T_matmul_local[4];
  __shared__ float left_matrix_shared[36];
  __shared__ float right_matrix_shared[36];
  T_matmul_local[0] = 0.000000e+00f;
  T_matmul_local[1] = 0.000000e+00f;
  T_matmul_local[2] = 0.000000e+00f;
  T_matmul_local[3] = 0.000000e+00f;
  for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 3; ++ax0_ax1_fused_outer_outer) {
    *(float3*)(left_matrix_shared + ((ax0_ax1_fused_outer_outer * 12) + (((int)threadIdx.x) * 3))) = *(float3*)(left_matrix + ((ax0_ax1_fused_outer_outer * 12) + (((int)threadIdx.x) * 3)));
  }
  for (int ax0_ax1_fused_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_1 < 9; ++ax0_ax1_fused_outer_outer_1) {
    right_matrix_shared[((ax0_ax1_fused_outer_outer_1 * 4) + ((int)threadIdx.x))] = right_matrix[((ax0_ax1_fused_outer_outer_1 * 4) + ((int)threadIdx.x))];
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 9; ++k_outer_inner) {
    T_matmul_local[0] = (T_matmul_local[0] + (left_matrix_shared[k_outer_inner] * right_matrix_shared[((k_outer_inner * 4) + ((int)threadIdx.x))]));
    T_matmul_local[1] = (T_matmul_local[1] + (left_matrix_shared[(k_outer_inner + 9)] * right_matrix_shared[((k_outer_inner * 4) + ((int)threadIdx.x))]));
    T_matmul_local[2] = (T_matmul_local[2] + (left_matrix_shared[(k_outer_inner + 18)] * right_matrix_shared[((k_outer_inner * 4) + ((int)threadIdx.x))]));
    T_matmul_local[3] = (T_matmul_local[3] + (left_matrix_shared[(k_outer_inner + 27)] * right_matrix_shared[((k_outer_inner * 4) + ((int)threadIdx.x))]));
  }
  T_matmul[((int)threadIdx.x)] = T_matmul_local[0];
  T_matmul[(((int)threadIdx.x) + 4)] = T_matmul_local[1];
  T_matmul[(((int)threadIdx.x) + 8)] = T_matmul_local[2];
  T_matmul[(((int)threadIdx.x) + 12)] = T_matmul_local[3];
}

