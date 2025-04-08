extern "C" __global__ void __launch_bounds__(64) default_function_kernel_1(float* __restrict__ T_mod, float* __restrict__ ph_0) {
  T_mod[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = fmodf(atanf(ph_0[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))]), ph_0[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(4) default_function_kernel_2(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))] = asinhf(atanf(ph_0[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))]));
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = fabsf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(45) default_function_kernel_3(float* __restrict__ compute, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  float T_batch_matmul_NN[8];
  __shared__ float ph_3_shared[90];
  for (int i_inner_init = 0; i_inner_init < 2; ++i_inner_init) {
    T_batch_matmul_NN[i_inner_init] = 0.000000e+00f;
    T_batch_matmul_NN[(i_inner_init + 2)] = 0.000000e+00f;
    T_batch_matmul_NN[(i_inner_init + 4)] = 0.000000e+00f;
    T_batch_matmul_NN[(i_inner_init + 6)] = 0.000000e+00f;
  }
  for (int k_outer_outer = 0; k_outer_outer < 4; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_outer_outer) {
      ph_3_shared[((ax0_ax1_fused_ax2_fused_outer_outer * 45) + ((int)threadIdx.x))] = ph_3[(((((((ax0_ax1_fused_ax2_fused_outer_outer * 9) + (((int)threadIdx.x) / 5)) >> 1) * 40) + (k_outer_outer * 10)) + ((((((int)threadIdx.x) / 5) + ax0_ax1_fused_ax2_fused_outer_outer) & 1) * 5)) + (((int)threadIdx.x) % 5))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 2; ++k_inner) {
      for (int i_inner = 0; i_inner < 2; ++i_inner) {
        T_batch_matmul_NN[i_inner] = (T_batch_matmul_NN[i_inner] + (ph_0[(((((((int)threadIdx.x) / 5) * 64) + (i_inner * 8)) + (k_outer_outer * 2)) + k_inner)] * ph_3_shared[((((((int)threadIdx.x) / 5) * 10) + (k_inner * 5)) + (((int)threadIdx.x) % 5))]));
        T_batch_matmul_NN[(i_inner + 2)] = (T_batch_matmul_NN[(i_inner + 2)] + (ph_0[((((((((int)threadIdx.x) / 5) * 64) + (i_inner * 8)) + (k_outer_outer * 2)) + k_inner) + 16)] * ph_3_shared[((((((int)threadIdx.x) / 5) * 10) + (k_inner * 5)) + (((int)threadIdx.x) % 5))]));
        T_batch_matmul_NN[(i_inner + 4)] = (T_batch_matmul_NN[(i_inner + 4)] + (ph_0[((((((((int)threadIdx.x) / 5) * 64) + (i_inner * 8)) + (k_outer_outer * 2)) + k_inner) + 32)] * ph_3_shared[((((((int)threadIdx.x) / 5) * 10) + (k_inner * 5)) + (((int)threadIdx.x) % 5))]));
        T_batch_matmul_NN[(i_inner + 6)] = (T_batch_matmul_NN[(i_inner + 6)] + (ph_0[((((((((int)threadIdx.x) / 5) * 64) + (i_inner * 8)) + (k_outer_outer * 2)) + k_inner) + 48)] * ph_3_shared[((((((int)threadIdx.x) / 5) * 10) + (k_inner * 5)) + (((int)threadIdx.x) % 5))]));
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 2; ++i1_inner) {
    compute[((((((int)threadIdx.x) / 5) * 40) + (i1_inner * 5)) + (((int)threadIdx.x) % 5))] = atanhf(T_batch_matmul_NN[i1_inner]);
    compute[(((((((int)threadIdx.x) / 5) * 40) + (i1_inner * 5)) + (((int)threadIdx.x) % 5)) + 10)] = atanhf(T_batch_matmul_NN[(i1_inner + 2)]);
    compute[(((((((int)threadIdx.x) / 5) * 40) + (i1_inner * 5)) + (((int)threadIdx.x) % 5)) + 20)] = atanhf(T_batch_matmul_NN[(i1_inner + 4)]);
    compute[(((((((int)threadIdx.x) / 5) * 40) + (i1_inner * 5)) + (((int)threadIdx.x) % 5)) + 30)] = atanhf(T_batch_matmul_NN[(i1_inner + 6)]);
  }
}

