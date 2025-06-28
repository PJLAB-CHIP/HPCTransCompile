extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = ceilf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_1(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = __sinf(acosf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]));
}

extern "C" __global__ void __launch_bounds__(10) default_function_kernel_2(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_5) {
  float T_batch_matmul_NN_local[2];
  __shared__ float ph_5_shared[80];
  T_batch_matmul_NN_local[0] = 0.000000e+00f;
  T_batch_matmul_NN_local[1] = 0.000000e+00f;
  for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_outer_outer) {
    ph_5_shared[((ax0_ax1_fused_ax2_fused_outer_outer * 10) + ((int)threadIdx.x))] = ph_5[((((((int)blockIdx.x) / 5) * 80) + (ax0_ax1_fused_ax2_fused_outer_outer * 10)) + ((int)threadIdx.x))];
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
    T_batch_matmul_NN_local[0] = (T_batch_matmul_NN_local[0] + (ph_0[(((((((int)blockIdx.x) / 5) * 80) + ((((int)threadIdx.x) / 5) * 20)) + ((((int)blockIdx.x) % 5) * 4)) + k_outer_inner)] * ph_5_shared[((((((int)threadIdx.x) / 5) * 20) + (k_outer_inner * 5)) + (((int)threadIdx.x) % 5))]));
    T_batch_matmul_NN_local[1] = (T_batch_matmul_NN_local[1] + (ph_0[((((((((int)blockIdx.x) / 5) * 80) + ((((int)threadIdx.x) / 5) * 20)) + ((((int)blockIdx.x) % 5) * 4)) + k_outer_inner) + 40)] * ph_5_shared[(((((((int)threadIdx.x) / 5) * 20) + (k_outer_inner * 5)) + (((int)threadIdx.x) % 5)) + 40)]));
  }
  T_batch_matmul_NN[(((((((int)blockIdx.x) / 5) * 100) + ((((int)threadIdx.x) / 5) * 25)) + ((((int)blockIdx.x) % 5) * 5)) + (((int)threadIdx.x) % 5))] = T_batch_matmul_NN_local[0];
  T_batch_matmul_NN[((((((((int)blockIdx.x) / 5) * 100) + ((((int)threadIdx.x) / 5) * 25)) + ((((int)blockIdx.x) % 5) * 5)) + (((int)threadIdx.x) % 5)) + 50)] = T_batch_matmul_NN_local[1];
}

