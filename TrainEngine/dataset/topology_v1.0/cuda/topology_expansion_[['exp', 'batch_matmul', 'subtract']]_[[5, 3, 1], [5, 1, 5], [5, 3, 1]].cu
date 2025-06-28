extern "C" __global__ void __launch_bounds__(32) default_function_kernel_2(float* __restrict__ T_subtract, float* __restrict__ ph_0, float* __restrict__ ph_5) {
  T_subtract[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] - ph_5[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = __expf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(40) default_function_kernel_1(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  float T_batch_matmul_NN_local[2];
  __shared__ float ph_3_shared[200];
  for (int b_c_outer_inner_init = 0; b_c_outer_inner_init < 2; ++b_c_outer_inner_init) {
    T_batch_matmul_NN_local[b_c_outer_inner_init] = 0.000000e+00f;
  }
  for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 3; ++ax0_ax1_fused_ax2_fused_outer_outer) {
    for (int ax0_ax1_fused_ax2_fused_inner_s = 0; ax0_ax1_fused_ax2_fused_inner_s < 2; ++ax0_ax1_fused_ax2_fused_inner_s) {
      if (((ax0_ax1_fused_ax2_fused_outer_outer * 2) + (((int)threadIdx.x) / 20)) < 5) {
        ph_3_shared[(((ax0_ax1_fused_ax2_fused_outer_outer * 80) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_inner_s)] = ph_3[(((ax0_ax1_fused_ax2_fused_outer_outer * 80) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_inner_s)];
      }
    }
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 5; ++k_outer_inner) {
    for (int b_c_outer_inner = 0; b_c_outer_inner < 2; ++b_c_outer_inner) {
      T_batch_matmul_NN_local[b_c_outer_inner] = (T_batch_matmul_NN_local[b_c_outer_inner] + (ph_0[((((((((int)threadIdx.x) / 10) * 80) + (b_c_outer_inner * 40)) + (((int)blockIdx.x) * 10)) + (((((int)threadIdx.x) % 10) / 5) * 5)) + k_outer_inner)] * ph_3_shared[(((((((int)threadIdx.x) / 10) * 50) + (b_c_outer_inner * 25)) + (k_outer_inner * 5)) + (((int)threadIdx.x) % 5))]));
    }
  }
  for (int b_inner = 0; b_inner < 2; ++b_inner) {
    T_batch_matmul_NN[(((((((int)threadIdx.x) / 10) * 80) + (b_inner * 40)) + (((int)blockIdx.x) * 10)) + (((int)threadIdx.x) % 10))] = T_batch_matmul_NN_local[b_inner];
  }
}

