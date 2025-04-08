extern "C" __global__ void default_function_kernel(float* __restrict__ T_matmul, float* __restrict__ left_matrix, float* __restrict__ right_matrix) {
  float T_matmul_local[11];
  __shared__ float left_matrix_shared[22];
  __shared__ float right_matrix_shared[2];
  T_matmul_local[0] = 0.000000e+00f;
  T_matmul_local[1] = 0.000000e+00f;
  T_matmul_local[2] = 0.000000e+00f;
  T_matmul_local[3] = 0.000000e+00f;
  T_matmul_local[4] = 0.000000e+00f;
  T_matmul_local[5] = 0.000000e+00f;
  T_matmul_local[6] = 0.000000e+00f;
  T_matmul_local[7] = 0.000000e+00f;
  T_matmul_local[8] = 0.000000e+00f;
  T_matmul_local[9] = 0.000000e+00f;
  T_matmul_local[10] = 0.000000e+00f;
  for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 22; ++ax0_ax1_fused_outer_outer) {
    left_matrix_shared[ax0_ax1_fused_outer_outer] = left_matrix[ax0_ax1_fused_outer_outer];
  }
  for (int ax0_ax1_fused_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_1 < 2; ++ax0_ax1_fused_outer_outer_1) {
    right_matrix_shared[ax0_ax1_fused_outer_outer_1] = right_matrix[((ax0_ax1_fused_outer_outer_1 * 11) + ((int)blockIdx.x))];
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
    T_matmul_local[0] = (T_matmul_local[0] + (left_matrix_shared[k_outer_inner] * right_matrix_shared[k_outer_inner]));
    T_matmul_local[1] = (T_matmul_local[1] + (left_matrix_shared[(k_outer_inner + 2)] * right_matrix_shared[k_outer_inner]));
    T_matmul_local[2] = (T_matmul_local[2] + (left_matrix_shared[(k_outer_inner + 4)] * right_matrix_shared[k_outer_inner]));
    T_matmul_local[3] = (T_matmul_local[3] + (left_matrix_shared[(k_outer_inner + 6)] * right_matrix_shared[k_outer_inner]));
    T_matmul_local[4] = (T_matmul_local[4] + (left_matrix_shared[(k_outer_inner + 8)] * right_matrix_shared[k_outer_inner]));
    T_matmul_local[5] = (T_matmul_local[5] + (left_matrix_shared[(k_outer_inner + 10)] * right_matrix_shared[k_outer_inner]));
    T_matmul_local[6] = (T_matmul_local[6] + (left_matrix_shared[(k_outer_inner + 12)] * right_matrix_shared[k_outer_inner]));
    T_matmul_local[7] = (T_matmul_local[7] + (left_matrix_shared[(k_outer_inner + 14)] * right_matrix_shared[k_outer_inner]));
    T_matmul_local[8] = (T_matmul_local[8] + (left_matrix_shared[(k_outer_inner + 16)] * right_matrix_shared[k_outer_inner]));
    T_matmul_local[9] = (T_matmul_local[9] + (left_matrix_shared[(k_outer_inner + 18)] * right_matrix_shared[k_outer_inner]));
    T_matmul_local[10] = (T_matmul_local[10] + (left_matrix_shared[(k_outer_inner + 20)] * right_matrix_shared[k_outer_inner]));
  }
  T_matmul[((int)blockIdx.x)] = T_matmul_local[0];
  T_matmul[(((int)blockIdx.x) + 11)] = T_matmul_local[1];
  T_matmul[(((int)blockIdx.x) + 22)] = T_matmul_local[2];
  T_matmul[(((int)blockIdx.x) + 33)] = T_matmul_local[3];
  T_matmul[(((int)blockIdx.x) + 44)] = T_matmul_local[4];
  T_matmul[(((int)blockIdx.x) + 55)] = T_matmul_local[5];
  T_matmul[(((int)blockIdx.x) + 66)] = T_matmul_local[6];
  T_matmul[(((int)blockIdx.x) + 77)] = T_matmul_local[7];
  T_matmul[(((int)blockIdx.x) + 88)] = T_matmul_local[8];
  T_matmul[(((int)blockIdx.x) + 99)] = T_matmul_local[9];
  T_matmul[(((int)blockIdx.x) + 110)] = T_matmul_local[10];
}

