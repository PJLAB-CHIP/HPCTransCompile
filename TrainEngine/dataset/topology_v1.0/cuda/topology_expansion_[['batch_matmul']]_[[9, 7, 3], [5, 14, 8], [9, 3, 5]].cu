extern "C" __global__ void __launch_bounds__(10) default_function_kernel(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  float T_batch_matmul_NN_local[2];
  __shared__ float ph_0_shared[40];
  __shared__ float ph_3_shared[8];
  T_batch_matmul_NN_local[0] = 0.000000e+00f;
  T_batch_matmul_NN_local[1] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 2; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_outer_outer) {
      ph_0_shared[((ax0_ax1_fused_ax2_fused_outer_outer * 10) + ((int)threadIdx.x))] = ph_0[((((((((int)blockIdx.x) / 5) * 80) + (ax0_ax1_fused_ax2_fused_outer_outer * 20)) + ((((int)threadIdx.x) >> 1) * 4)) + (k_outer_outer * 2)) + (((int)threadIdx.x) & 1))];
    }
    if (((int)threadIdx.x) < 8) {
      ph_3_shared[((int)threadIdx.x)] = ph_3[((((((((int)blockIdx.x) / 5) * 80) + ((((int)threadIdx.x) >> 1) * 20)) + (k_outer_outer * 10)) + ((((int)threadIdx.x) & 1) * 5)) + (((int)blockIdx.x) % 5))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 2; ++k_inner) {
      T_batch_matmul_NN_local[0] = (T_batch_matmul_NN_local[0] + (ph_0_shared[((((int)threadIdx.x) * 2) + k_inner)] * ph_3_shared[(((((int)threadIdx.x) / 5) * 2) + k_inner)]));
      T_batch_matmul_NN_local[1] = (T_batch_matmul_NN_local[1] + (ph_0_shared[(((((int)threadIdx.x) * 2) + k_inner) + 20)] * ph_3_shared[((((((int)threadIdx.x) / 5) * 2) + k_inner) + 4)]));
    }
  }
  T_batch_matmul_NN[((((((int)blockIdx.x) / 5) * 100) + (((int)threadIdx.x) * 5)) + (((int)blockIdx.x) % 5))] = T_batch_matmul_NN_local[0];
  T_batch_matmul_NN[(((((((int)blockIdx.x) / 5) * 100) + (((int)threadIdx.x) * 5)) + (((int)blockIdx.x) % 5)) + 50)] = T_batch_matmul_NN_local[1];
}

