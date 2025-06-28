extern "C" __global__ void __launch_bounds__(32) default_function_kernel_2(float* __restrict__ T_multiply, float* __restrict__ ph_0) {
  T_multiply[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (fmodf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], asinf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])) * ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_1(float* __restrict__ T_multiply, float* __restrict__ ph_0) {
  T_multiply[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (acoshf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]) * ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(4) default_function_kernel_3(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_8) {
  float T_batch_matmul_NN_local[2];
  __shared__ float ph_8_shared[20];
  for (int b_c_outer_inner_init = 0; b_c_outer_inner_init < 2; ++b_c_outer_inner_init) {
    T_batch_matmul_NN_local[b_c_outer_inner_init] = 0.000000e+00f;
  }
  for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_outer_outer) {
    ph_8_shared[((ax0_ax1_fused_ax2_fused_outer_outer * 4) + ((int)threadIdx.x))] = ph_8[((((((int)blockIdx.x) >> 2) * 20) + (ax0_ax1_fused_ax2_fused_outer_outer * 4)) + ((int)threadIdx.x))];
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 5; ++k_outer_inner) {
    for (int b_c_outer_inner = 0; b_c_outer_inner < 2; ++b_c_outer_inner) {
      T_batch_matmul_NN_local[b_c_outer_inner] = (T_batch_matmul_NN_local[b_c_outer_inner] + (fmodf(ph_0[(((((((((int)blockIdx.x) >> 2) * 160) + ((((int)threadIdx.x) >> 1) * 80)) + (b_c_outer_inner * 40)) + ((((int)blockIdx.x) & 3) * 10)) + ((((int)threadIdx.x) & 1) * 5)) + k_outer_inner)], asinf(ph_0[(((((((((int)blockIdx.x) >> 2) * 160) + ((((int)threadIdx.x) >> 1) * 80)) + (b_c_outer_inner * 40)) + ((((int)blockIdx.x) & 3) * 10)) + ((((int)threadIdx.x) & 1) * 5)) + k_outer_inner)])) * ph_8_shared[((((((int)threadIdx.x) >> 1) * 10) + (b_c_outer_inner * 5)) + k_outer_inner)]));
    }
  }
  for (int b_inner = 0; b_inner < 2; ++b_inner) {
    T_batch_matmul_NN[((((((((int)blockIdx.x) >> 2) * 32) + ((((int)threadIdx.x) >> 1) * 16)) + (b_inner * 8)) + ((((int)blockIdx.x) & 3) * 2)) + (((int)threadIdx.x) & 1))] = T_batch_matmul_NN_local[b_inner];
  }
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = fabsf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

