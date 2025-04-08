extern "C" __global__ void __launch_bounds__(72) default_function_kernel_1(float* __restrict__ T_subtract, float* __restrict__ ph_0) {
  float T_batch_matmul_NN[1];
  __shared__ float compute_shared[576];
  T_batch_matmul_NN[0] = 0.000000e+00f;
  for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_outer_outer) {
    compute_shared[((ax0_ax1_fused_ax2_fused_outer_outer * 72) + ((int)threadIdx.x))] = acoshf(ph_0[((ax0_ax1_fused_ax2_fused_outer_outer * 72) + ((int)threadIdx.x))]);
  }
  __syncthreads();
  for (int k_inner = 0; k_inner < 8; ++k_inner) {
    T_batch_matmul_NN[0] = (T_batch_matmul_NN[0] + (compute_shared[((((int)threadIdx.x) * 8) + k_inner)] * ph_0[((((((int)threadIdx.x) >> 3) * 64) + (k_inner * 8)) + ((int)blockIdx.x))]));
  }
  T_subtract[((((int)threadIdx.x) * 8) + ((int)blockIdx.x))] = (ph_0[((((int)threadIdx.x) * 8) + ((int)blockIdx.x))] - T_batch_matmul_NN[0]);
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_2(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = atanf(fabsf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]));
}

extern "C" __global__ void __launch_bounds__(64) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = asinhf(ph_0[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))]);
}

