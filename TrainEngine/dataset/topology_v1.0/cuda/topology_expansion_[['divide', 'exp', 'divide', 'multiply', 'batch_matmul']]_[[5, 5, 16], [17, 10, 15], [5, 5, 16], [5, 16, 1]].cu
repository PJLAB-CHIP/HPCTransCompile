extern "C" __global__ void __launch_bounds__(4) default_function_kernel_2(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_8) {
  float T_batch_matmul_NN_local[16];
  __shared__ float ph_8_shared[40];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 8; ++i_c_outer_inner_init) {
    T_batch_matmul_NN_local[i_c_outer_inner_init] = 0.000000e+00f;
    T_batch_matmul_NN_local[(i_c_outer_inner_init + 8)] = 0.000000e+00f;
  }
  for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 10; ++ax0_ax1_fused_ax2_fused_outer_outer) {
    ph_8_shared[((ax0_ax1_fused_ax2_fused_outer_outer * 4) + ((int)threadIdx.x))] = ph_8[((ax0_ax1_fused_ax2_fused_outer_outer * 4) + ((int)threadIdx.x))];
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 5; ++k_outer_inner) {
    for (int i_c_outer_inner = 0; i_c_outer_inner < 8; ++i_c_outer_inner) {
      T_batch_matmul_NN_local[i_c_outer_inner] = (T_batch_matmul_NN_local[i_c_outer_inner] + ((ph_0[(((((int)threadIdx.x) * 40) + (i_c_outer_inner * 5)) + k_outer_inner)] / __expf(ph_0[(((((int)threadIdx.x) * 40) + (i_c_outer_inner * 5)) + k_outer_inner)])) * ph_8_shared[((((int)threadIdx.x) * 5) + k_outer_inner)]));
      T_batch_matmul_NN_local[(i_c_outer_inner + 8)] = (T_batch_matmul_NN_local[(i_c_outer_inner + 8)] + ((ph_0[((((((int)threadIdx.x) * 40) + (i_c_outer_inner * 5)) + k_outer_inner) + 160)] / __expf(ph_0[((((((int)threadIdx.x) * 40) + (i_c_outer_inner * 5)) + k_outer_inner) + 160)])) * ph_8_shared[(((((int)threadIdx.x) * 5) + k_outer_inner) + 20)]));
    }
  }
  for (int i_inner = 0; i_inner < 8; ++i_inner) {
    T_batch_matmul_NN[((((int)threadIdx.x) * 8) + i_inner)] = T_batch_matmul_NN_local[i_inner];
    T_batch_matmul_NN[(((((int)threadIdx.x) * 8) + i_inner) + 32)] = T_batch_matmul_NN_local[(i_inner + 8)];
  }
}

extern "C" __global__ void __launch_bounds__(64) default_function_kernel(float* __restrict__ T_divide, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  T_divide[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = (ph_0[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] / ph_3[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_1(float* __restrict__ T_multiply, float* __restrict__ ph_0) {
  T_multiply[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = ((ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] / __expf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])) * ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

