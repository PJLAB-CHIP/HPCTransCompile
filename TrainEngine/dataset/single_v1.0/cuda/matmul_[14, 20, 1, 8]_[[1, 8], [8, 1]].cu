extern "C" __global__ void default_function_kernel(float* __restrict__ T_matmul, float* __restrict__ left_matrix, float* __restrict__ right_matrix) {
  T_matmul[0] = 0.000000e+00f;
  for (int k = 0; k < 8; ++k) {
    T_matmul[0] = (T_matmul[0] + (left_matrix[k] * right_matrix[k]));
  }
}

