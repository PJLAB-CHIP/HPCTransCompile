extern "C" __global__ void __launch_bounds__(20) default_function_kernel_1(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_5) {
  float T_batch_matmul_NN_local[2];
  __shared__ float ph_5_shared[32];
  T_batch_matmul_NN_local[0] = 0.000000e+00f;
  T_batch_matmul_NN_local[1] = 0.000000e+00f;
  for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_outer_outer) {
    if (((ax0_ax1_fused_ax2_fused_outer_outer * 5) + (((int)threadIdx.x) >> 2)) < 8) {
      ph_5_shared[((ax0_ax1_fused_ax2_fused_outer_outer * 20) + ((int)threadIdx.x))] = ph_5[(((ax0_ax1_fused_ax2_fused_outer_outer * 100) + (((int)threadIdx.x) * 5)) + ((int)blockIdx.x))];
    }
  }
  __syncthreads();
  for (int k_inner = 0; k_inner < 4; ++k_inner) {
    T_batch_matmul_NN_local[0] = (T_batch_matmul_NN_local[0] + (ph_0[((((int)threadIdx.x) * 4) + k_inner)] * ph_5_shared[(((((int)threadIdx.x) / 5) * 4) + k_inner)]));
    T_batch_matmul_NN_local[1] = (T_batch_matmul_NN_local[1] + (ph_0[(((((int)threadIdx.x) * 4) + k_inner) + 80)] * ph_5_shared[((((((int)threadIdx.x) / 5) * 4) + k_inner) + 16)]));
  }
  T_batch_matmul_NN[((((int)threadIdx.x) * 5) + ((int)blockIdx.x))] = T_batch_matmul_NN_local[0];
  T_batch_matmul_NN[(((((int)threadIdx.x) * 5) + ((int)blockIdx.x)) + 100)] = T_batch_matmul_NN_local[1];
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ T_subtract, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  T_subtract[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] - ph_3[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

