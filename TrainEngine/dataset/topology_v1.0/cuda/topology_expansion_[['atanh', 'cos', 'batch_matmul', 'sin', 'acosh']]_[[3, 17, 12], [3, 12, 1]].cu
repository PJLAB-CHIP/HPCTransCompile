extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 3)) < 9) {
    compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = atanhf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void default_function_kernel_1(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_4) {
  float T_batch_matmul_NN_local[9];
  __shared__ float compute_shared[18];
  __shared__ float ph_4_shared[2];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 9; ++i_c_outer_inner_init) {
    T_batch_matmul_NN_local[i_c_outer_inner_init] = 0.000000e+00f;
  }
  for (int k_outer_outer = 0; k_outer_outer < 4; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 18; ++ax0_ax1_fused_ax2_fused_outer_outer) {
      compute_shared[ax0_ax1_fused_ax2_fused_outer_outer] = __cosf(ph_0[((((ax0_ax1_fused_ax2_fused_outer_outer >> 1) * 8) + (k_outer_outer * 2)) + (ax0_ax1_fused_ax2_fused_outer_outer & 1))]);
    }
    *(float2*)(ph_4_shared + 0) = *(float2*)(ph_4 + (k_outer_outer * 2));
    __syncthreads();
    for (int i_c_outer_inner = 0; i_c_outer_inner < 9; ++i_c_outer_inner) {
      for (int k_inner = 0; k_inner < 2; ++k_inner) {
        T_batch_matmul_NN_local[i_c_outer_inner] = (T_batch_matmul_NN_local[i_c_outer_inner] + (compute_shared[((i_c_outer_inner * 2) + k_inner)] * ph_4_shared[k_inner]));
      }
    }
  }
  for (int i_inner = 0; i_inner < 9; ++i_inner) {
    T_batch_matmul_NN[i_inner] = T_batch_matmul_NN_local[i_inner];
  }
}

extern "C" __global__ void __launch_bounds__(64) default_function_kernel_2(float* __restrict__ compute, float* __restrict__ ph_0) {
  if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 3)) < 9) {
    compute[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = acoshf(__sinf(ph_0[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))]));
  }
}

