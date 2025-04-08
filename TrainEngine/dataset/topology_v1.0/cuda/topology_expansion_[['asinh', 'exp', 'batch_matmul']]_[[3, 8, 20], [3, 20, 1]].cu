extern "C" __global__ void __launch_bounds__(8) default_function_kernel_1(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_4) {
  float T_batch_matmul_NN_local[4];
  __shared__ float compute_shared[160];
  __shared__ float ph_4_shared[40];
  for (int b_c_inner_init = 0; b_c_inner_init < 2; ++b_c_inner_init) {
    T_batch_matmul_NN_local[b_c_inner_init] = 0.000000e+00f;
    T_batch_matmul_NN_local[(b_c_inner_init + 2)] = 0.000000e+00f;
  }
  for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 20; ++ax0_ax1_fused_ax2_fused_outer_outer) {
    compute_shared[((ax0_ax1_fused_ax2_fused_outer_outer * 8) + ((int)threadIdx.x))] = __expf(ph_0[((((((ax0_ax1_fused_ax2_fused_outer_outer * 2) + (((int)threadIdx.x) >> 2)) / 5) * 40) + (((int)blockIdx.x) * 20)) + (((ax0_ax1_fused_ax2_fused_outer_outer * 8) + ((int)threadIdx.x)) % 20))]);
  }
  for (int ax0_ax1_fused_ax2_fused_outer_outer_1 = 0; ax0_ax1_fused_ax2_fused_outer_outer_1 < 5; ++ax0_ax1_fused_ax2_fused_outer_outer_1) {
    ph_4_shared[((ax0_ax1_fused_ax2_fused_outer_outer_1 * 8) + ((int)threadIdx.x))] = ph_4[((ax0_ax1_fused_ax2_fused_outer_outer_1 * 8) + ((int)threadIdx.x))];
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 5; ++k_outer_inner) {
    for (int b_c_inner = 0; b_c_inner < 2; ++b_c_inner) {
      T_batch_matmul_NN_local[b_c_inner] = (T_batch_matmul_NN_local[b_c_inner] + (compute_shared[(((((((int)threadIdx.x) >> 1) * 40) + (b_c_inner * 20)) + ((((int)threadIdx.x) & 1) * 5)) + k_outer_inner)] * ph_4_shared[((((((int)threadIdx.x) >> 1) * 10) + (b_c_inner * 5)) + k_outer_inner)]));
      T_batch_matmul_NN_local[(b_c_inner + 2)] = (T_batch_matmul_NN_local[(b_c_inner + 2)] + (compute_shared[((((((((int)threadIdx.x) >> 1) * 40) + (b_c_inner * 20)) + ((((int)threadIdx.x) & 1) * 5)) + k_outer_inner) + 10)] * ph_4_shared[((((((int)threadIdx.x) >> 1) * 10) + (b_c_inner * 5)) + k_outer_inner)]));
    }
  }
  for (int b_inner = 0; b_inner < 2; ++b_inner) {
    T_batch_matmul_NN[(((((((int)threadIdx.x) >> 1) * 16) + (b_inner * 8)) + (((int)blockIdx.x) * 4)) + (((int)threadIdx.x) & 1))] = T_batch_matmul_NN_local[b_inner];
    T_batch_matmul_NN[((((((((int)threadIdx.x) >> 1) * 16) + (b_inner * 8)) + (((int)blockIdx.x) * 4)) + (((int)threadIdx.x) & 1)) + 2)] = T_batch_matmul_NN_local[(b_inner + 2)];
  }
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = asinhf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

