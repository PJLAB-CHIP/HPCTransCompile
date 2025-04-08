extern "C" __global__ void default_function_kernel_1(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_8) {
  float T_batch_matmul_NN_local[8];
  __shared__ float ph_8_shared[20];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 2; ++i_c_outer_inner_init) {
    for (int b_c_inner_init = 0; b_c_inner_init < 4; ++b_c_inner_init) {
      T_batch_matmul_NN_local[((b_c_inner_init * 2) + i_c_outer_inner_init)] = 0.000000e+00f;
    }
  }
  for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 20; ++ax0_ax1_fused_ax2_fused_outer_outer) {
    ph_8_shared[ax0_ax1_fused_ax2_fused_outer_outer] = ph_8[(((((int)blockIdx.x) >> 2) * 20) + ax0_ax1_fused_ax2_fused_outer_outer)];
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 5; ++k_outer_inner) {
    for (int i_c_outer_inner = 0; i_c_outer_inner < 2; ++i_c_outer_inner) {
      for (int b_c_inner = 0; b_c_inner < 4; ++b_c_inner) {
        T_batch_matmul_NN_local[((b_c_inner * 2) + i_c_outer_inner)] = (T_batch_matmul_NN_local[((b_c_inner * 2) + i_c_outer_inner)] + (fmodf(ph_0[((((((((int)blockIdx.x) >> 2) * 160) + (b_c_inner * 40)) + ((((int)blockIdx.x) & 3) * 10)) + (i_c_outer_inner * 5)) + k_outer_inner)], (ph_0[((((((((int)blockIdx.x) >> 2) * 160) + (b_c_inner * 40)) + ((((int)blockIdx.x) & 3) * 10)) + (i_c_outer_inner * 5)) + k_outer_inner)] + ((ph_0[((((((((int)blockIdx.x) >> 2) * 160) + (b_c_inner * 40)) + ((((int)blockIdx.x) & 3) * 10)) + (i_c_outer_inner * 5)) + k_outer_inner)] * fabsf(ph_0[((((((((int)blockIdx.x) >> 2) * 160) + (b_c_inner * 40)) + ((((int)blockIdx.x) & 3) * 10)) + (i_c_outer_inner * 5)) + k_outer_inner)])) / ph_0[((((((((int)blockIdx.x) >> 2) * 160) + (b_c_inner * 40)) + ((((int)blockIdx.x) & 3) * 10)) + (i_c_outer_inner * 5)) + k_outer_inner)]))) * ph_8_shared[((b_c_inner * 5) + k_outer_inner)]));
      }
    }
  }
  for (int b_inner = 0; b_inner < 4; ++b_inner) {
    for (int i_inner = 0; i_inner < 2; ++i_inner) {
      T_batch_matmul_NN[(((((((int)blockIdx.x) >> 2) * 32) + (b_inner * 8)) + ((((int)blockIdx.x) & 3) * 2)) + i_inner)] = T_batch_matmul_NN_local[((b_inner * 2) + i_inner)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(10) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 10) + ((int)threadIdx.x))] = asinhf(fmodf(ph_0[((((int)blockIdx.x) * 10) + ((int)threadIdx.x))], (ph_0[((((int)blockIdx.x) * 10) + ((int)threadIdx.x))] + ((ph_0[((((int)blockIdx.x) * 10) + ((int)threadIdx.x))] * fabsf(ph_0[((((int)blockIdx.x) * 10) + ((int)threadIdx.x))])) / ph_0[((((int)blockIdx.x) * 10) + ((int)threadIdx.x))]))));
}

