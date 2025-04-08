extern "C" __global__ void __launch_bounds__(40) default_function_kernel_1(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  float T_batch_matmul_NN_local[1];
  __shared__ float ph_3_shared[50];
  T_batch_matmul_NN_local[0] = 0.000000e+00f;
  for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_outer_outer) {
    if (((ax0_ax1_fused_ax2_fused_outer_outer * 4) + (((int)threadIdx.x) / 10)) < 5) {
      ph_3_shared[((ax0_ax1_fused_ax2_fused_outer_outer * 40) + ((int)threadIdx.x))] = ph_3[((((((int)blockIdx.x) >> 1) * 50) + (ax0_ax1_fused_ax2_fused_outer_outer * 40)) + ((int)threadIdx.x))];
    }
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 5; ++k_outer_inner) {
    T_batch_matmul_NN_local[0] = (T_batch_matmul_NN_local[0] + (ph_0[((((((((int)blockIdx.x) >> 1) * 80) + ((((int)threadIdx.x) / 20) * 40)) + ((((int)blockIdx.x) & 1) * 20)) + (((((int)threadIdx.x) % 20) / 5) * 5)) + k_outer_inner)] * ph_3_shared[((((((int)threadIdx.x) / 20) * 25) + (k_outer_inner * 5)) + (((int)threadIdx.x) % 5))]));
  }
  T_batch_matmul_NN[(((((((int)blockIdx.x) >> 1) * 80) + ((((int)threadIdx.x) / 20) * 40)) + ((((int)blockIdx.x) & 1) * 20)) + (((int)threadIdx.x) % 20))] = T_batch_matmul_NN_local[0];
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = ceilf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

