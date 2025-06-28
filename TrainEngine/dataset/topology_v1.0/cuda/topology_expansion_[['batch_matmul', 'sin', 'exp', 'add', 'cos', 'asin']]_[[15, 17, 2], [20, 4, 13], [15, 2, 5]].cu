extern "C" __global__ void __launch_bounds__(32) default_function_kernel_1(float* __restrict__ T_add, float* __restrict__ ph_0) {
  T_add[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (__expf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]) + ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = __sinf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_2(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = __cosf(__expf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]));
}

extern "C" __global__ void __launch_bounds__(45) default_function_kernel_3(float* __restrict__ compute, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  float T_batch_matmul_NN[8];
  __shared__ float ph_3_shared[180];
  for (int i_inner_init = 0; i_inner_init < 8; ++i_inner_init) {
    T_batch_matmul_NN[i_inner_init] = 0.000000e+00f;
  }
  for (int k_outer_outer = 0; k_outer_outer < 2; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 4; ++ax0_ax1_fused_ax2_fused_outer_outer) {
      ph_3_shared[((ax0_ax1_fused_ax2_fused_outer_outer * 45) + ((int)threadIdx.x))] = ph_3[(((((((ax0_ax1_fused_ax2_fused_outer_outer * 9) + (((int)threadIdx.x) / 5)) >> 2) * 40) + (k_outer_outer * 20)) + ((((((int)threadIdx.x) / 5) + ax0_ax1_fused_ax2_fused_outer_outer) & 3) * 5)) + (((int)threadIdx.x) % 5))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 4; ++k_inner) {
      for (int i_inner = 0; i_inner < 8; ++i_inner) {
        T_batch_matmul_NN[i_inner] = (T_batch_matmul_NN[i_inner] + (ph_0[(((((((int)threadIdx.x) / 5) * 64) + (i_inner * 8)) + (k_outer_outer * 4)) + k_inner)] * ph_3_shared[((((((int)threadIdx.x) / 5) * 20) + (k_inner * 5)) + (((int)threadIdx.x) % 5))]));
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 8; ++i1_inner) {
    compute[((((((int)threadIdx.x) / 5) * 40) + (i1_inner * 5)) + (((int)threadIdx.x) % 5))] = asinf(T_batch_matmul_NN[i1_inner]);
  }
}

