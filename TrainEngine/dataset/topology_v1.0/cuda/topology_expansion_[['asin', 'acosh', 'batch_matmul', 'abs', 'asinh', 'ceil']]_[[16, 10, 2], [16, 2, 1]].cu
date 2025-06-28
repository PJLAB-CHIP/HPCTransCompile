extern "C" __global__ void default_function_kernel_1(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_4) {
  float T_batch_matmul_NN_local[3];
  __shared__ float compute_shared[12];
  __shared__ float ph_4_shared[4];
  for (int i_c_inner_init = 0; i_c_inner_init < 3; ++i_c_inner_init) {
    T_batch_matmul_NN_local[i_c_inner_init] = 0.000000e+00f;
  }
  for (int k_outer_outer = 0; k_outer_outer < 2; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 12; ++ax0_ax1_fused_ax2_fused_outer_outer) {
      compute_shared[ax0_ax1_fused_ax2_fused_outer_outer] = acoshf(ph_0[((((((int)blockIdx.x) * 24) + ((ax0_ax1_fused_ax2_fused_outer_outer >> 2) * 8)) + (k_outer_outer * 4)) + (ax0_ax1_fused_ax2_fused_outer_outer & 3))]);
    }
    for (int ax0_ax1_fused_ax2_fused_outer_outer_1 = 0; ax0_ax1_fused_ax2_fused_outer_outer_1 < 4; ++ax0_ax1_fused_ax2_fused_outer_outer_1) {
      ph_4_shared[ax0_ax1_fused_ax2_fused_outer_outer_1] = ph_4[((k_outer_outer * 4) + ax0_ax1_fused_ax2_fused_outer_outer_1)];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 4; ++k_inner) {
      for (int i_c_inner = 0; i_c_inner < 3; ++i_c_inner) {
        T_batch_matmul_NN_local[i_c_inner] = (T_batch_matmul_NN_local[i_c_inner] + (compute_shared[((i_c_inner * 4) + k_inner)] * ph_4_shared[k_inner]));
      }
    }
  }
  for (int i_inner = 0; i_inner < 3; ++i_inner) {
    T_batch_matmul_NN[((((int)blockIdx.x) * 3) + i_inner)] = T_batch_matmul_NN_local[i_inner];
  }
}

extern "C" __global__ void __launch_bounds__(64) default_function_kernel_3(float* __restrict__ compute, float* __restrict__ ph_0) {
  if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 3)) < 9) {
    compute[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = ceilf(fabsf(ph_0[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))]));
  }
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_2(float* __restrict__ compute, float* __restrict__ ph_0) {
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 3)) < 9) {
    compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = asinhf(fabsf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]));
  }
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 3)) < 9) {
    compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = asinf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
  }
}

