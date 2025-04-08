extern "C" __global__ void __launch_bounds__(25) default_function_kernel_3(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_3, float* __restrict__ ph_9) {
  float T_batch_matmul_NN_local[8];
  __shared__ float T_divide_shared[160];
  __shared__ float ph_9_shared[160];
  for (int b_c_outer_inner_init = 0; b_c_outer_inner_init < 4; ++b_c_outer_inner_init) {
    T_batch_matmul_NN_local[b_c_outer_inner_init] = 0.000000e+00f;
    T_batch_matmul_NN_local[(b_c_outer_inner_init + 4)] = 0.000000e+00f;
  }
  for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 7; ++ax0_ax1_fused_ax2_fused_outer_outer) {
    if (((ax0_ax1_fused_ax2_fused_outer_outer * 5) + (((int)threadIdx.x) / 5)) < 32) {
      T_divide_shared[((ax0_ax1_fused_ax2_fused_outer_outer * 25) + ((int)threadIdx.x))] = (ph_0[((ax0_ax1_fused_ax2_fused_outer_outer * 25) + ((int)threadIdx.x))] / ph_3[((ax0_ax1_fused_ax2_fused_outer_outer * 25) + ((int)threadIdx.x))]);
    }
  }
  for (int ax0_ax1_fused_ax2_fused_outer_outer_1 = 0; ax0_ax1_fused_ax2_fused_outer_outer_1 < 7; ++ax0_ax1_fused_ax2_fused_outer_outer_1) {
    if (((ax0_ax1_fused_ax2_fused_outer_outer_1 * 5) + (((int)threadIdx.x) / 5)) < 32) {
      ph_9_shared[((ax0_ax1_fused_ax2_fused_outer_outer_1 * 25) + ((int)threadIdx.x))] = ph_9[((ax0_ax1_fused_ax2_fused_outer_outer_1 * 25) + ((int)threadIdx.x))];
    }
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
    for (int b_c_outer_inner = 0; b_c_outer_inner < 4; ++b_c_outer_inner) {
      T_batch_matmul_NN_local[b_c_outer_inner] = (T_batch_matmul_NN_local[b_c_outer_inner] + (T_divide_shared[(((b_c_outer_inner * 20) + ((((int)threadIdx.x) / 5) * 4)) + k_outer_inner)] * ph_9_shared[(((b_c_outer_inner * 20) + (k_outer_inner * 5)) + (((int)threadIdx.x) % 5))]));
      T_batch_matmul_NN_local[(b_c_outer_inner + 4)] = (T_batch_matmul_NN_local[(b_c_outer_inner + 4)] + (T_divide_shared[((((b_c_outer_inner * 20) + ((((int)threadIdx.x) / 5) * 4)) + k_outer_inner) + 80)] * ph_9_shared[((((b_c_outer_inner * 20) + (k_outer_inner * 5)) + (((int)threadIdx.x) % 5)) + 80)]));
    }
  }
  for (int b_inner = 0; b_inner < 4; ++b_inner) {
    T_batch_matmul_NN[((b_inner * 25) + ((int)threadIdx.x))] = T_batch_matmul_NN_local[b_inner];
    T_batch_matmul_NN[(((b_inner * 25) + ((int)threadIdx.x)) + 100)] = T_batch_matmul_NN_local[(b_inner + 4)];
  }
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = fabsf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_1(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = acosf(acosf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]));
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_2(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = ceilf(acosf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]));
}

