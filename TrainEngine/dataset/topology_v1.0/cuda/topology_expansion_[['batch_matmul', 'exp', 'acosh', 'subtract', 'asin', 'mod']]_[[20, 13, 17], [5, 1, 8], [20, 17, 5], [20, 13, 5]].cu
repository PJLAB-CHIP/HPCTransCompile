extern "C" __global__ void __launch_bounds__(32) default_function_kernel_2(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = asinf(acoshf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]));
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = __expf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(20) default_function_kernel_3(float* __restrict__ T_mod, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  float T_batch_matmul_NN[16];
  __shared__ float ph_3_shared[200];
  for (int b_outer_inner_init = 0; b_outer_inner_init < 2; ++b_outer_inner_init) {
    for (int i_outer_inner_init = 0; i_outer_inner_init < 8; ++i_outer_inner_init) {
      T_batch_matmul_NN[((b_outer_inner_init * 8) + i_outer_inner_init)] = 0.000000e+00f;
    }
  }
  for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 10; ++ax0_ax1_fused_ax2_fused_outer_outer) {
    ph_3_shared[((ax0_ax1_fused_ax2_fused_outer_outer * 20) + ((int)threadIdx.x))] = ph_3[((ax0_ax1_fused_ax2_fused_outer_outer * 20) + ((int)threadIdx.x))];
  }
  __syncthreads();
  for (int b_outer_inner = 0; b_outer_inner < 2; ++b_outer_inner) {
    for (int i_outer_inner = 0; i_outer_inner < 8; ++i_outer_inner) {
      for (int k_inner = 0; k_inner < 5; ++k_inner) {
        T_batch_matmul_NN[((b_outer_inner * 8) + i_outer_inner)] = (T_batch_matmul_NN[((b_outer_inner * 8) + i_outer_inner)] + (ph_0[(((((((int)threadIdx.x) / 5) * 80) + (b_outer_inner * 40)) + (i_outer_inner * 5)) + k_inner)] * ph_3_shared[(((((((int)threadIdx.x) / 5) * 50) + (b_outer_inner * 25)) + (k_inner * 5)) + (((int)threadIdx.x) % 5))]));
      }
    }
  }
  for (int ax0_inner = 0; ax0_inner < 2; ++ax0_inner) {
    for (int ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
      T_mod[(((((((int)threadIdx.x) / 5) * 80) + (ax0_inner * 40)) + (ax1_inner * 5)) + (((int)threadIdx.x) % 5))] = fmodf(T_batch_matmul_NN[((ax0_inner * 8) + ax1_inner)], ph_0[(((((((int)threadIdx.x) / 5) * 80) + (ax0_inner * 40)) + (ax1_inner * 5)) + (((int)threadIdx.x) % 5))]);
    }
  }
}

extern "C" __global__ void __launch_bounds__(64) default_function_kernel_1(float* __restrict__ T_subtract, float* __restrict__ ph_0) {
  T_subtract[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = (acoshf(ph_0[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))]) - ph_0[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))]);
}

