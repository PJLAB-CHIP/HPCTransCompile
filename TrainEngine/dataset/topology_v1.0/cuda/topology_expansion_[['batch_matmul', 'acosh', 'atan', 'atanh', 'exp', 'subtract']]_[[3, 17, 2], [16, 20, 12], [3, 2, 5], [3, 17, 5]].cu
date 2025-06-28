extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = acoshf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(8) default_function_kernel_3(float* __restrict__ T_subtract, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  float T_batch_matmul_NN[8];
  __shared__ float ph_3_shared[40];
  for (int i_inner_init = 0; i_inner_init < 4; ++i_inner_init) {
    T_batch_matmul_NN[i_inner_init] = 0.000000e+00f;
    T_batch_matmul_NN[(i_inner_init + 4)] = 0.000000e+00f;
  }
  for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_outer_outer) {
    ph_3_shared[((ax0_ax1_fused_ax2_fused_outer_outer * 8) + ((int)threadIdx.x))] = ph_3[(((ax0_ax1_fused_ax2_fused_outer_outer * 40) + (((int)threadIdx.x) * 5)) + ((int)blockIdx.x))];
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 5; ++k_outer_inner) {
    for (int i_inner = 0; i_inner < 4; ++i_inner) {
      T_batch_matmul_NN[i_inner] = (T_batch_matmul_NN[i_inner] + (ph_0[(((((int)threadIdx.x) * 40) + (i_inner * 5)) + k_outer_inner)] * ph_3_shared[((((int)threadIdx.x) * 5) + k_outer_inner)]));
      T_batch_matmul_NN[(i_inner + 4)] = (T_batch_matmul_NN[(i_inner + 4)] + (ph_0[((((((int)threadIdx.x) * 40) + (i_inner * 5)) + k_outer_inner) + 20)] * ph_3_shared[((((int)threadIdx.x) * 5) + k_outer_inner)]));
    }
  }
  for (int ax1_inner = 0; ax1_inner < 4; ++ax1_inner) {
    T_subtract[(((((int)threadIdx.x) * 40) + (ax1_inner * 5)) + ((int)blockIdx.x))] = (T_batch_matmul_NN[ax1_inner] - ph_0[(((((int)threadIdx.x) * 40) + (ax1_inner * 5)) + ((int)blockIdx.x))]);
    T_subtract[((((((int)threadIdx.x) * 40) + (ax1_inner * 5)) + ((int)blockIdx.x)) + 20)] = (T_batch_matmul_NN[(ax1_inner + 4)] - ph_0[((((((int)threadIdx.x) * 40) + (ax1_inner * 5)) + ((int)blockIdx.x)) + 20)]);
  }
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_2(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = __expf(atanf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]));
}

extern "C" __global__ void __launch_bounds__(64) default_function_kernel_1(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = atanhf(atanf(ph_0[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))]));
}

