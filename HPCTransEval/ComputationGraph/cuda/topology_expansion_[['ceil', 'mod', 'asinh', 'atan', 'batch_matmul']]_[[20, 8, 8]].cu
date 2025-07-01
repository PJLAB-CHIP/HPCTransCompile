extern "C" __global__ void __launch_bounds__(40) default_function_kernel_2(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_6) {
  float T_batch_matmul_NN_local[2];
  __shared__ float ph_6_shared[50];
  for (int b_c_inner_init = 0; b_c_inner_init < 2; ++b_c_inner_init) {
    T_batch_matmul_NN_local[b_c_inner_init] = 0.000000e+00f;
  }
  for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_outer_outer) {
    if (((ax0_ax1_fused_ax2_fused_outer_outer * 4) + (((int)threadIdx.x) / 10)) < 5) {
      ph_6_shared[((ax0_ax1_fused_ax2_fused_outer_outer * 40) + ((int)threadIdx.x))] = ph_6[(((((int)blockIdx.x) * 50) + (ax0_ax1_fused_ax2_fused_outer_outer * 40)) + ((int)threadIdx.x))];
    }
  }
  __syncthreads();
  for (int k_inner = 0; k_inner < 5; ++k_inner) {
    for (int b_c_inner = 0; b_c_inner < 2; ++b_c_inner) {
      T_batch_matmul_NN_local[b_c_inner] = (T_batch_matmul_NN_local[b_c_inner] + (ph_0[((((((int)blockIdx.x) * 80) + (b_c_inner * 40)) + ((((int)threadIdx.x) / 5) * 5)) + k_inner)] * ph_6_shared[(((b_c_inner * 25) + (k_inner * 5)) + (((int)threadIdx.x) % 5))]));
    }
  }
  for (int b_inner = 0; b_inner < 2; ++b_inner) {
    T_batch_matmul_NN[(((((int)blockIdx.x) * 80) + (b_inner * 40)) + ((int)threadIdx.x))] = T_batch_matmul_NN_local[b_inner];
  }
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = asinhf(fmodf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], ceilf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])));
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_1(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = atanf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

