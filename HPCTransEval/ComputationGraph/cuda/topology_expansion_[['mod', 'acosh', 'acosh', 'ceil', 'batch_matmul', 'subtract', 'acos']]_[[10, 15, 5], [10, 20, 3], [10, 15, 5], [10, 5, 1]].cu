extern "C" __global__ void __launch_bounds__(32) default_function_kernel_4(float* __restrict__ compute, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = acosf(fmodf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], ph_3[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]));
}

extern "C" __global__ void default_function_kernel_3(float* __restrict__ T_subtract, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  T_subtract[((int)blockIdx.x)] = (fmodf(ph_0[((int)blockIdx.x)], ph_3[((int)blockIdx.x)]) - ph_0[((int)blockIdx.x)]);
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_1(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = ceilf(acoshf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]));
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = acoshf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(18) default_function_kernel_2(float* __restrict__ T_batch_matmul_NN, float* __restrict__ ph_0) {
  float T_batch_matmul_NN_local[32];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 4; ++i_c_outer_inner_init) {
    for (int j_c_outer_inner_init = 0; j_c_outer_inner_init < 4; ++j_c_outer_inner_init) {
      for (int j_c_inner_init = 0; j_c_inner_init < 2; ++j_c_inner_init) {
        T_batch_matmul_NN_local[(((i_c_outer_inner_init * 8) + (j_c_outer_inner_init * 2)) + j_c_inner_init)] = 0.000000e+00f;
      }
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 2; ++k_outer_outer) {
    for (int i_c_outer_inner = 0; i_c_outer_inner < 4; ++i_c_outer_inner) {
      for (int j_c_outer_inner = 0; j_c_outer_inner < 4; ++j_c_outer_inner) {
        for (int k_inner = 0; k_inner < 4; ++k_inner) {
          for (int j_c_inner = 0; j_c_inner < 2; ++j_c_inner) {
            T_batch_matmul_NN_local[(((i_c_outer_inner * 8) + (j_c_outer_inner * 2)) + j_c_inner)] = (T_batch_matmul_NN_local[(((i_c_outer_inner * 8) + (j_c_outer_inner * 2)) + j_c_inner)] + (acoshf(ph_0[((((((int)threadIdx.x) * 32) + (i_c_outer_inner * 8)) + (k_outer_outer * 4)) + k_inner)]) * ph_0[((((((((int)threadIdx.x) >> 1) * 64) + (k_outer_outer * 32)) + (k_inner * 8)) + (j_c_outer_inner * 2)) + j_c_inner)]));
          }
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 4; ++i_inner) {
    for (int j_inner = 0; j_inner < 8; ++j_inner) {
      T_batch_matmul_NN[(((((int)threadIdx.x) * 32) + (i_inner * 8)) + j_inner)] = T_batch_matmul_NN_local[((i_inner * 8) + j_inner)];
    }
  }
}

