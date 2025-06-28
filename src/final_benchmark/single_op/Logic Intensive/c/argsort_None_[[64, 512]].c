
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
    Input CUDA Code: extern "C" __global__ void default_function_kernel_2(float* __restrict__ argsort_gpu, float* __restrict__ argsort_gpu_v0, float* __restrict__ argsort_gpu_v2, float* __restrict__ argsort_gpu_v3, int64_t cse_var_1, int64_t i_0) {
  int64_t first[1];
  int64_t last[1];
  int64_t first_1[1];
  int64_t last_1[1];
  if ((((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z))) < (int64_t)512) {
    if (i_0 == (int64_t)0) {
      first[0] = max((int64_t)0, ((((((int64_t)2 << cse_var_1) >> (int64_t)1) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z)))) + (((int64_t)((int)threadIdx.x)) * (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1))) - (((int64_t)2 << cse_var_1) * (((int64_t)((int)blockIdx.z)) + (int64_t)1))));
      last[0] = min((((int64_t)((int)threadIdx.x)) * (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1)), (((int64_t)2 << cse_var_1) >> (int64_t)1));
      while ((first[0] < last[0])) {
        if (argsort_gpu_v0[(((((int64_t)((int)blockIdx.y)) * (int64_t)512) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z)))) + ((first[0] + last[0]) >> (int64_t)1))] <= argsort_gpu_v0[((((((((int64_t)((int)blockIdx.y)) * (int64_t)512) + (((int64_t)2 << cse_var_1) >> (int64_t)1)) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z)))) + (((int64_t)((int)threadIdx.x)) * (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1))) - ((first[0] + last[0]) >> (int64_t)1)) - (int64_t)1)]) {
          first[0] = (((first[0] + last[0]) >> (int64_t)1) + (int64_t)1);
        } else {
          last[0] = ((first[0] + last[0]) >> (int64_t)1);
        }
      }
      first[0] = ((((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z))) + first[0]);
      last[0] = ((((((int64_t)2 << cse_var_1) >> (int64_t)1) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z)))) + (((int64_t)((int)threadIdx.x)) * (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1))) - last[0]);
      for (int i_1 = 0; i_1 < ((int)min((((int64_t)2 << cse_var_1) - (((int64_t)((int)threadIdx.x)) * (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1))), (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1))); ++i_1) {
        if ((((first[0] < ((((int64_t)2 << cse_var_1) >> (int64_t)1) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z))))) && (first[0] < (int64_t)512)) && (last[0] < (((int64_t)2 << cse_var_1) * (((int64_t)((int)blockIdx.z)) + (int64_t)1)))) && (last[0] < (int64_t)512)) {
          if (argsort_gpu_v0[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + first[0])] <= argsort_gpu_v0[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + last[0])]) {
            argsort_gpu_v2[((((((int64_t)((int)blockIdx.y)) * (int64_t)512) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z)))) + (((int64_t)((int)threadIdx.x)) * (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1))) + ((int64_t)i_1))] = argsort_gpu_v0[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + first[0])];
            argsort_gpu_v3[((((((int64_t)((int)blockIdx.y)) * (int64_t)512) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z)))) + (((int64_t)((int)threadIdx.x)) * (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1))) + ((int64_t)i_1))] = argsort_gpu[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + first[0])];
            first[0] = (first[0] + (int64_t)1);
          } else {
            argsort_gpu_v2[((((((int64_t)((int)blockIdx.y)) * (int64_t)512) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z)))) + (((int64_t)((int)threadIdx.x)) * (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1))) + ((int64_t)i_1))] = argsort_gpu_v0[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + last[0])];
            argsort_gpu_v3[((((((int64_t)((int)blockIdx.y)) * (int64_t)512) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z)))) + (((int64_t)((int)threadIdx.x)) * (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1))) + ((int64_t)i_1))] = argsort_gpu[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + last[0])];
            last[0] = (last[0] + (int64_t)1);
          }
        } else {
          if ((first[0] < ((((int64_t)2 << cse_var_1) >> (int64_t)1) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z))))) && (first[0] < (int64_t)512)) {
            argsort_gpu_v2[((((((int64_t)((int)blockIdx.y)) * (int64_t)512) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z)))) + (((int64_t)((int)threadIdx.x)) * (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1))) + ((int64_t)i_1))] = argsort_gpu_v0[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + first[0])];
            argsort_gpu_v3[((((((int64_t)((int)blockIdx.y)) * (int64_t)512) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z)))) + (((int64_t)((int)threadIdx.x)) * (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1))) + ((int64_t)i_1))] = argsort_gpu[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + first[0])];
            first[0] = (first[0] + (int64_t)1);
          } else {
            argsort_gpu_v2[((((((int64_t)((int)blockIdx.y)) * (int64_t)512) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z)))) + (((int64_t)((int)threadIdx.x)) * (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1))) + ((int64_t)i_1))] = argsort_gpu_v0[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + last[0])];
            argsort_gpu_v3[((((((int64_t)((int)blockIdx.y)) * (int64_t)512) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z)))) + (((int64_t)((int)threadIdx.x)) * (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1))) + ((int64_t)i_1))] = argsort_gpu[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + last[0])];
            last[0] = (last[0] + (int64_t)1);
          }
        }
      }
    } else {
      first_1[0] = max((int64_t)0, ((min(((((int64_t)2 << cse_var_1) >> (int64_t)1) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z)))), (int64_t)512) + (((int64_t)((int)threadIdx.x)) * (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1))) - (int64_t)512));
      last_1[0] = min((((int64_t)((int)threadIdx.x)) * (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1)), min((((int64_t)2 << cse_var_1) >> (int64_t)1), ((int64_t)512 - (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z))))));
      while ((first_1[0] < last_1[0])) {
        if (argsort_gpu_v2[(((((int64_t)((int)blockIdx.y)) * (int64_t)512) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z)))) + ((first_1[0] + last_1[0]) >> (int64_t)1))] <= argsort_gpu_v2[(((((((int64_t)((int)blockIdx.y)) * (int64_t)512) + min(((((int64_t)2 << cse_var_1) >> (int64_t)1) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z)))), (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1))) - ((first_1[0] + last_1[0]) >> (int64_t)1)) - (int64_t)1)]) {
          first_1[0] = (((first_1[0] + last_1[0]) >> (int64_t)1) + (int64_t)1);
        } else {
          last_1[0] = ((first_1[0] + last_1[0]) >> (int64_t)1);
        }
      }
      first_1[0] = ((((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z))) + first_1[0]);
      last_1[0] = ((min(((((int64_t)2 << cse_var_1) >> (int64_t)1) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z)))), (int64_t)512) + (((int64_t)((int)threadIdx.x)) * (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1))) - last_1[0]);
      for (int i_2 = 0; i_2 < ((int)min((((int64_t)512 - (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z)))) - (((int64_t)((int)threadIdx.x)) * (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1))), (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1))); ++i_2) {
        if (((first_1[0] < ((((int64_t)2 << cse_var_1) >> (int64_t)1) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z))))) && (first_1[0] < (int64_t)512)) && (last_1[0] < (int64_t)512)) {
          if (argsort_gpu_v2[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + first_1[0])] <= argsort_gpu_v2[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + last_1[0])]) {
            argsort_gpu_v0[((((((int64_t)((int)blockIdx.y)) * (int64_t)512) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z)))) + (((int64_t)((int)threadIdx.x)) * (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1))) + ((int64_t)i_2))] = argsort_gpu_v2[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + first_1[0])];
            argsort_gpu[((((((int64_t)((int)blockIdx.y)) * (int64_t)512) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z)))) + (((int64_t)((int)threadIdx.x)) * (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1))) + ((int64_t)i_2))] = argsort_gpu_v3[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + first_1[0])];
            first_1[0] = (first_1[0] + (int64_t)1);
          } else {
            argsort_gpu_v0[((((((int64_t)((int)blockIdx.y)) * (int64_t)512) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z)))) + (((int64_t)((int)threadIdx.x)) * (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1))) + ((int64_t)i_2))] = argsort_gpu_v2[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + last_1[0])];
            argsort_gpu[((((((int64_t)((int)blockIdx.y)) * (int64_t)512) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z)))) + (((int64_t)((int)threadIdx.x)) * (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1))) + ((int64_t)i_2))] = argsort_gpu_v3[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + last_1[0])];
            last_1[0] = (last_1[0] + (int64_t)1);
          }
        } else {
          if ((first_1[0] < ((((int64_t)2 << cse_var_1) >> (int64_t)1) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z))))) && (first_1[0] < (int64_t)512)) {
            argsort_gpu_v0[((((((int64_t)((int)blockIdx.y)) * (int64_t)512) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z)))) + (((int64_t)((int)threadIdx.x)) * (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1))) + ((int64_t)i_2))] = argsort_gpu_v2[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + first_1[0])];
            argsort_gpu[((((((int64_t)((int)blockIdx.y)) * (int64_t)512) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z)))) + (((int64_t)((int)threadIdx.x)) * (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1))) + ((int64_t)i_2))] = argsort_gpu_v3[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + first_1[0])];
            first_1[0] = (first_1[0] + (int64_t)1);
          } else {
            argsort_gpu_v0[((((((int64_t)((int)blockIdx.y)) * (int64_t)512) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z)))) + (((int64_t)((int)threadIdx.x)) * (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1))) + ((int64_t)i_2))] = argsort_gpu_v2[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + last_1[0])];
            argsort_gpu[((((((int64_t)((int)blockIdx.y)) * (int64_t)512) + (((int64_t)2 << cse_var_1) * ((int64_t)((int)blockIdx.z)))) + (((int64_t)((int)threadIdx.x)) * (((((((int64_t)((int)((int64_t)2 << cse_var_1))) >= (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) >= (int64_t)0)) || ((((int64_t)((int)((int64_t)2 << cse_var_1))) < (int64_t)0) && (((((int64_t)2 << cse_var_1) - (int64_t)1) % ((int64_t)((int)((int64_t)2 << cse_var_1)))) <= (int64_t)0))) ? ((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) : (((((int64_t)2 << cse_var_1) - (int64_t)1) / ((int64_t)((int)((int64_t)2 << cse_var_1)))) - (int64_t)1)) + (int64_t)1))) + ((int64_t)i_2))] = argsort_gpu_v3[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + last_1[0])];
            last_1[0] = (last_1[0] + (int64_t)1);
          }
        }
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ argsort_gpu, float* __restrict__ argsort_gpu_v0, float* __restrict__ data) {
  if (((int)threadIdx.x) < 512) {
    argsort_gpu_v0[((((int)blockIdx.y) * 512) + ((int)threadIdx.x))] = data[((((int)blockIdx.y) * 512) + ((int)threadIdx.x))];
    argsort_gpu[((((int)blockIdx.y) * 512) + ((int)threadIdx.x))] = ((float)((int)threadIdx.x));
  }
}

extern "C" __global__ void __launch_bounds__(64) default_function_kernel_1(float* __restrict__ argsort_gpu, float* __restrict__ argsort_gpu_v0, float* __restrict__ argsort_gpu_v2, float* __restrict__ argsort_gpu_v3) {
  __shared__ float temp_keys_swap[128];
  __shared__ float temp_values_swap[128];
  float temp_cond1[1];
  float temp_cond2[1];
  float temp_keys[1];
  float temp_values[1];
  for (int i = 0; i < 2; ++i) {
    temp_keys_swap[((((int)threadIdx.x) * 2) + i)] = argsort_gpu_v0[((((((int)blockIdx.y) * 512) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) * 2)) + i)];
    temp_values_swap[((((int)threadIdx.x) * 2) + i)] = argsort_gpu[((((((int)blockIdx.y) * 512) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) * 2)) + i)];
  }
  __syncthreads();
  for (int j = 0; j < 128; ++j) {
    if (((((int)threadIdx.x) * 2) + (j & 1)) < 127) {
      temp_cond1[0] = temp_keys_swap[((((int)threadIdx.x) * 2) + (j & 1))];
      temp_cond2[0] = temp_keys_swap[(((((int)threadIdx.x) * 2) + (j & 1)) + 1)];
      if (temp_cond2[0] < temp_cond1[0]) {
        temp_keys[0] = temp_keys_swap[((((int)threadIdx.x) * 2) + (j & 1))];
        temp_keys_swap[((((int)threadIdx.x) * 2) + (j & 1))] = temp_keys_swap[(((((int)threadIdx.x) * 2) + (j & 1)) + 1)];
        temp_keys_swap[(((((int)threadIdx.x) * 2) + (j & 1)) + 1)] = temp_keys[0];
        temp_values[0] = temp_values_swap[((((int)threadIdx.x) * 2) + (j & 1))];
        temp_values_swap[((((int)threadIdx.x) * 2) + (j & 1))] = temp_values_swap[(((((int)threadIdx.x) * 2) + (j & 1)) + 1)];
        temp_values_swap[(((((int)threadIdx.x) * 2) + (j & 1)) + 1)] = temp_values[0];
      }
    }
    __syncthreads();
  }
  for (int k = 0; k < 2; ++k) {
    argsort_gpu_v0[((((((int)blockIdx.y) * 512) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) * 2)) + k)] = temp_keys_swap[((((int)threadIdx.x) * 2) + k)];
    argsort_gpu_v2[((((((int)blockIdx.y) * 512) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) * 2)) + k)] = temp_keys_swap[((((int)threadIdx.x) * 2) + k)];
    argsort_gpu[((((((int)blockIdx.y) * 512) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) * 2)) + k)] = temp_values_swap[((((int)threadIdx.x) * 2) + k)];
    argsort_gpu_v3[((((((int)blockIdx.y) * 512) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) * 2)) + k)] = temp_values_swap[((((int)threadIdx.x) * 2) + k)];
  }
}


    Input Tensor Shape: [[64, 512]]
    Print only a single C function implementation, ending with the comment '|End-of-Code|'.
    