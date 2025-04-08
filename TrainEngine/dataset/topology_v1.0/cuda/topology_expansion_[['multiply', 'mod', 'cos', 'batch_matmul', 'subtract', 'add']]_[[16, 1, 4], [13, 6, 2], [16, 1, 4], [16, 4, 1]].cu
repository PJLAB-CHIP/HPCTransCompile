extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ T_mod, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  T_mod[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = fmodf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], ph_3[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(2) default_function_kernel_1(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0, float* __restrict__ ph_7) {
  float T_batch_matmul_NN_local[32];
  __shared__ float compute_shared[64];
  __shared__ float ph_7_shared[8];
  for (int b_c_outer_inner_init = 0; b_c_outer_inner_init < 4; ++b_c_outer_inner_init) {
    for (int b_c_inner_init = 0; b_c_inner_init < 2; ++b_c_inner_init) {
      T_batch_matmul_NN_local[((b_c_outer_inner_init * 2) + b_c_inner_init)] = 0.000000e+00f;
      T_batch_matmul_NN_local[(((b_c_outer_inner_init * 2) + b_c_inner_init) + 8)] = 0.000000e+00f;
      T_batch_matmul_NN_local[(((b_c_outer_inner_init * 2) + b_c_inner_init) + 16)] = 0.000000e+00f;
      T_batch_matmul_NN_local[(((b_c_outer_inner_init * 2) + b_c_inner_init) + 24)] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 5; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 32; ++ax0_ax1_fused_ax2_fused_outer_outer) {
      compute_shared[((ax0_ax1_fused_ax2_fused_outer_outer * 2) + ((int)threadIdx.x))] = __cosf(ph_0[(((ax0_ax1_fused_ax2_fused_outer_outer * 10) + (((int)threadIdx.x) * 5)) + k_outer_outer)]);
    }
    for (int ax0_ax1_fused_ax2_fused_outer_outer_1 = 0; ax0_ax1_fused_ax2_fused_outer_outer_1 < 4; ++ax0_ax1_fused_ax2_fused_outer_outer_1) {
      ph_7_shared[((ax0_ax1_fused_ax2_fused_outer_outer_1 * 2) + ((int)threadIdx.x))] = ph_7[(((ax0_ax1_fused_ax2_fused_outer_outer_1 * 10) + (((int)threadIdx.x) * 5)) + k_outer_outer)];
    }
    __syncthreads();
    for (int b_c_outer_inner = 0; b_c_outer_inner < 4; ++b_c_outer_inner) {
      for (int b_c_inner = 0; b_c_inner < 2; ++b_c_inner) {
        T_batch_matmul_NN_local[((b_c_outer_inner * 2) + b_c_inner)] = (T_batch_matmul_NN_local[((b_c_outer_inner * 2) + b_c_inner)] + (compute_shared[(((b_c_outer_inner * 16) + (b_c_inner * 8)) + ((int)threadIdx.x))] * ph_7_shared[((b_c_outer_inner * 2) + b_c_inner)]));
        T_batch_matmul_NN_local[(((b_c_outer_inner * 2) + b_c_inner) + 8)] = (T_batch_matmul_NN_local[(((b_c_outer_inner * 2) + b_c_inner) + 8)] + (compute_shared[((((b_c_outer_inner * 16) + (b_c_inner * 8)) + ((int)threadIdx.x)) + 2)] * ph_7_shared[((b_c_outer_inner * 2) + b_c_inner)]));
        T_batch_matmul_NN_local[(((b_c_outer_inner * 2) + b_c_inner) + 16)] = (T_batch_matmul_NN_local[(((b_c_outer_inner * 2) + b_c_inner) + 16)] + (compute_shared[((((b_c_outer_inner * 16) + (b_c_inner * 8)) + ((int)threadIdx.x)) + 4)] * ph_7_shared[((b_c_outer_inner * 2) + b_c_inner)]));
        T_batch_matmul_NN_local[(((b_c_outer_inner * 2) + b_c_inner) + 24)] = (T_batch_matmul_NN_local[(((b_c_outer_inner * 2) + b_c_inner) + 24)] + (compute_shared[((((b_c_outer_inner * 16) + (b_c_inner * 8)) + ((int)threadIdx.x)) + 6)] * ph_7_shared[((b_c_outer_inner * 2) + b_c_inner)]));
      }
    }
  }
  for (int b_inner = 0; b_inner < 8; ++b_inner) {
    T_batch_matmul_NN[((b_inner * 8) + ((int)threadIdx.x))] = T_batch_matmul_NN_local[b_inner];
    T_batch_matmul_NN[(((b_inner * 8) + ((int)threadIdx.x)) + 2)] = T_batch_matmul_NN_local[(b_inner + 8)];
    T_batch_matmul_NN[(((b_inner * 8) + ((int)threadIdx.x)) + 4)] = T_batch_matmul_NN_local[(b_inner + 16)];
    T_batch_matmul_NN[(((b_inner * 8) + ((int)threadIdx.x)) + 6)] = T_batch_matmul_NN_local[(b_inner + 24)];
  }
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_2(float* __restrict__ T_subtract, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  T_subtract[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = ((ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] * ph_3[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]) - ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_3(float* __restrict__ T_add, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  T_add[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = ((ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] * ph_3[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]) + ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

