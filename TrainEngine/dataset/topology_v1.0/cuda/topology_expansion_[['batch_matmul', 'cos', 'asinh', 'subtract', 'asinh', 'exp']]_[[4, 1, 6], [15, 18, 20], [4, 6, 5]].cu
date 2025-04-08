extern "C" __global__ void __launch_bounds__(120) default_function_kernel_3(float* __restrict__ compute, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  float T_batch_matmul_NN[3];
  __shared__ float ph_3_shared[360];
  for (int b_inner_init = 0; b_inner_init < 3; ++b_inner_init) {
    T_batch_matmul_NN[b_inner_init] = 0.000000e+00f;
  }
  for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 3; ++ax0_ax1_fused_ax2_fused_outer_outer) {
    ph_3_shared[((ax0_ax1_fused_ax2_fused_outer_outer * 120) + ((int)threadIdx.x))] = ph_3[((ax0_ax1_fused_ax2_fused_outer_outer * 120) + ((int)threadIdx.x))];
  }
  __syncthreads();
  for (int k_inner = 0; k_inner < 8; ++k_inner) {
    for (int b_inner = 0; b_inner < 3; ++b_inner) {
      T_batch_matmul_NN[b_inner] = (T_batch_matmul_NN[b_inner] + (ph_0[(((((((int)threadIdx.x) / 40) * 192) + (b_inner * 64)) + (((((int)threadIdx.x) % 40) / 5) * 8)) + k_inner)] * ph_3_shared[(((((((int)threadIdx.x) / 40) * 120) + (b_inner * 40)) + (k_inner * 5)) + (((int)threadIdx.x) % 5))]));
    }
  }
  for (int i0_inner = 0; i0_inner < 3; ++i0_inner) {
    compute[((((((int)threadIdx.x) / 40) * 120) + (i0_inner * 40)) + (((int)threadIdx.x) % 40))] = __expf(T_batch_matmul_NN[i0_inner]);
  }
}

extern "C" __global__ void __launch_bounds__(64) default_function_kernel_1(float* __restrict__ T_subtract, float* __restrict__ ph_0) {
  T_subtract[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = (asinhf(ph_0[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))]) - ph_0[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = __cosf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_2(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = asinhf(asinhf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]));
}

