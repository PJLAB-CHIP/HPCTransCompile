extern "C" __global__ void __launch_bounds__(32) default_function_kernel_3(float* __restrict__ T_add, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  T_add[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (fmodf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], ph_3[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]) + ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void default_function_kernel_2(float* __restrict__ T_multiply, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  T_multiply[((int)blockIdx.x)] = (fmodf(ph_0[((int)blockIdx.x)], ph_3[((int)blockIdx.x)]) * ph_0[((int)blockIdx.x)]);
}

extern "C" __global__ void __launch_bounds__(4) default_function_kernel_1(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_7) {
  float T_batch_matmul_NN_local[2];
  __shared__ float compute_shared[40];
  __shared__ float ph_7_shared[40];
  for (int b_c_inner_init = 0; b_c_inner_init < 2; ++b_c_inner_init) {
    T_batch_matmul_NN_local[b_c_inner_init] = 0.000000e+00f;
  }
  for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 10; ++ax0_ax1_fused_ax2_fused_outer_outer) {
    compute_shared[((ax0_ax1_fused_ax2_fused_outer_outer * 4) + ((int)threadIdx.x))] = acosf(ph_0[((((((ax0_ax1_fused_ax2_fused_outer_outer * 4) + ((int)threadIdx.x)) / 5) * 40) + (((int)blockIdx.x) * 5)) + (((ax0_ax1_fused_ax2_fused_outer_outer * 4) + ((int)threadIdx.x)) % 5))]);
  }
  for (int ax0_ax1_fused_ax2_fused_outer_outer_1 = 0; ax0_ax1_fused_ax2_fused_outer_outer_1 < 10; ++ax0_ax1_fused_ax2_fused_outer_outer_1) {
    ph_7_shared[((ax0_ax1_fused_ax2_fused_outer_outer_1 * 4) + ((int)threadIdx.x))] = ph_7[((ax0_ax1_fused_ax2_fused_outer_outer_1 * 4) + ((int)threadIdx.x))];
  }
  __syncthreads();
  for (int k_inner = 0; k_inner < 5; ++k_inner) {
    for (int b_c_inner = 0; b_c_inner < 2; ++b_c_inner) {
      T_batch_matmul_NN_local[b_c_inner] = (T_batch_matmul_NN_local[b_c_inner] + (compute_shared[(((((int)threadIdx.x) * 10) + (b_c_inner * 5)) + k_inner)] * ph_7_shared[(((((int)threadIdx.x) * 10) + (b_c_inner * 5)) + k_inner)]));
    }
  }
  for (int b_inner = 0; b_inner < 2; ++b_inner) {
    T_batch_matmul_NN[(((((int)threadIdx.x) * 16) + (b_inner * 8)) + ((int)blockIdx.x))] = T_batch_matmul_NN_local[b_inner];
  }
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = __expf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

