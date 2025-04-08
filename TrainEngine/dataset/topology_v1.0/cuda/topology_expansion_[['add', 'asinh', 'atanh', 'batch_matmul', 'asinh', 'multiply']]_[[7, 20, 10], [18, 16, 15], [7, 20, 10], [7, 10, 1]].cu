extern "C" __global__ void __launch_bounds__(32) default_function_kernel_2(float* __restrict__ compute, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = asinhf((ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + ph_3[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]));
}

extern "C" __global__ void __launch_bounds__(64) default_function_kernel_3(float* __restrict__ T_multiply, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  T_multiply[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = ((ph_0[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] + ph_3[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))]) * ph_0[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = asinhf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void default_function_kernel_1(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_7) {
  float T_batch_matmul_NN_local[32];
  __shared__ float compute_shared[160];
  __shared__ float ph_7_shared[40];
  for (int b_c_outer_inner_init = 0; b_c_outer_inner_init < 2; ++b_c_outer_inner_init) {
    for (int i_c_inner_init = 0; i_c_inner_init < 4; ++i_c_inner_init) {
      T_batch_matmul_NN_local[((b_c_outer_inner_init * 4) + i_c_inner_init)] = 0.000000e+00f;
      T_batch_matmul_NN_local[(((b_c_outer_inner_init * 4) + i_c_inner_init) + 8)] = 0.000000e+00f;
      T_batch_matmul_NN_local[(((b_c_outer_inner_init * 4) + i_c_inner_init) + 16)] = 0.000000e+00f;
      T_batch_matmul_NN_local[(((b_c_outer_inner_init * 4) + i_c_inner_init) + 24)] = 0.000000e+00f;
    }
  }
  for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 160; ++ax0_ax1_fused_ax2_fused_outer_outer) {
    compute_shared[ax0_ax1_fused_ax2_fused_outer_outer] = atanhf(ph_0[((((ax0_ax1_fused_ax2_fused_outer_outer / 20) * 40) + (((int)blockIdx.x) * 20)) + (ax0_ax1_fused_ax2_fused_outer_outer % 20))]);
  }
  for (int ax0_ax1_fused_ax2_fused_outer_outer_1 = 0; ax0_ax1_fused_ax2_fused_outer_outer_1 < 40; ++ax0_ax1_fused_ax2_fused_outer_outer_1) {
    ph_7_shared[ax0_ax1_fused_ax2_fused_outer_outer_1] = ph_7[ax0_ax1_fused_ax2_fused_outer_outer_1];
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 5; ++k_outer_inner) {
    for (int b_c_outer_inner = 0; b_c_outer_inner < 2; ++b_c_outer_inner) {
      for (int i_c_inner = 0; i_c_inner < 4; ++i_c_inner) {
        T_batch_matmul_NN_local[((b_c_outer_inner * 4) + i_c_inner)] = (T_batch_matmul_NN_local[((b_c_outer_inner * 4) + i_c_inner)] + (compute_shared[(((b_c_outer_inner * 20) + (i_c_inner * 5)) + k_outer_inner)] * ph_7_shared[((b_c_outer_inner * 5) + k_outer_inner)]));
        T_batch_matmul_NN_local[(((b_c_outer_inner * 4) + i_c_inner) + 8)] = (T_batch_matmul_NN_local[(((b_c_outer_inner * 4) + i_c_inner) + 8)] + (compute_shared[((((b_c_outer_inner * 20) + (i_c_inner * 5)) + k_outer_inner) + 40)] * ph_7_shared[(((b_c_outer_inner * 5) + k_outer_inner) + 10)]));
        T_batch_matmul_NN_local[(((b_c_outer_inner * 4) + i_c_inner) + 16)] = (T_batch_matmul_NN_local[(((b_c_outer_inner * 4) + i_c_inner) + 16)] + (compute_shared[((((b_c_outer_inner * 20) + (i_c_inner * 5)) + k_outer_inner) + 80)] * ph_7_shared[(((b_c_outer_inner * 5) + k_outer_inner) + 20)]));
        T_batch_matmul_NN_local[(((b_c_outer_inner * 4) + i_c_inner) + 24)] = (T_batch_matmul_NN_local[(((b_c_outer_inner * 4) + i_c_inner) + 24)] + (compute_shared[((((b_c_outer_inner * 20) + (i_c_inner * 5)) + k_outer_inner) + 120)] * ph_7_shared[(((b_c_outer_inner * 5) + k_outer_inner) + 30)]));
      }
    }
  }
  for (int b_inner = 0; b_inner < 2; ++b_inner) {
    for (int i_inner = 0; i_inner < 4; ++i_inner) {
      T_batch_matmul_NN[(((b_inner * 8) + (((int)blockIdx.x) * 4)) + i_inner)] = T_batch_matmul_NN_local[((b_inner * 4) + i_inner)];
      T_batch_matmul_NN[((((b_inner * 8) + (((int)blockIdx.x) * 4)) + i_inner) + 16)] = T_batch_matmul_NN_local[(((b_inner * 4) + i_inner) + 8)];
      T_batch_matmul_NN[((((b_inner * 8) + (((int)blockIdx.x) * 4)) + i_inner) + 32)] = T_batch_matmul_NN_local[(((b_inner * 4) + i_inner) + 16)];
      T_batch_matmul_NN[((((b_inner * 8) + (((int)blockIdx.x) * 4)) + i_inner) + 48)] = T_batch_matmul_NN_local[(((b_inner * 4) + i_inner) + 24)];
    }
  }
}

