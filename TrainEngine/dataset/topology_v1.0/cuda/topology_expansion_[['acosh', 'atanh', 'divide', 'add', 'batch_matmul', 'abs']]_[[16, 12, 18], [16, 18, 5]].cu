extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 3)) < 9) {
    compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = acoshf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_1(float* __restrict__ T_add, float* __restrict__ ph_0) {
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 3)) < 9) {
    T_add[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + (atanhf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]) / ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]));
  }
}

extern "C" __global__ void __launch_bounds__(5) default_function_kernel_2(float* __restrict__ compute, float* __restrict__ ph_0, float* __restrict__ ph_6) {
  float T_batch_matmul_NN[9];
  __shared__ float ph_6_shared[40];
  for (int i_inner_init = 0; i_inner_init < 9; ++i_inner_init) {
    T_batch_matmul_NN[i_inner_init] = 0.000000e+00f;
  }
  for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_outer_outer) {
    ph_6_shared[((ax0_ax1_fused_ax2_fused_outer_outer * 5) + ((int)threadIdx.x))] = ph_6[((ax0_ax1_fused_ax2_fused_outer_outer * 5) + ((int)threadIdx.x))];
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
    for (int k_inner = 0; k_inner < 4; ++k_inner) {
      for (int i_inner = 0; i_inner < 9; ++i_inner) {
        T_batch_matmul_NN[i_inner] = (T_batch_matmul_NN[i_inner] + (ph_0[(((i_inner * 8) + (k_outer_inner * 4)) + k_inner)] * ph_6_shared[(((k_outer_inner * 20) + (k_inner * 5)) + ((int)threadIdx.x))]));
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 9; ++i1_inner) {
    compute[((i1_inner * 5) + ((int)threadIdx.x))] = fabsf(T_batch_matmul_NN[i1_inner]);
  }
}

