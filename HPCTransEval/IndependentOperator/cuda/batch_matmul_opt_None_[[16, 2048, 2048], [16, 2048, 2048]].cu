extern "C" __global__ void __launch_bounds__(64) default_function_kernel(float* __restrict__ T_batch_matmul_NT, float* __restrict__ x, float* __restrict__ y) {
  float T_batch_matmul_NT_local[64];
  __shared__ float x_shared[512];
  __shared__ float y_shared[512];
  float x_shared_local[8];
  float y_shared_local[8];
  for (int i_c_init = 0; i_c_init < 8; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 8; ++j_c_init) {
      T_batch_matmul_NT_local[((i_c_init * 8) + j_c_init)] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 256; ++k_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
      x_shared[(((((int)threadIdx.y) * 64) + (ax1_inner * 8)) + ((int)threadIdx.x))] = x[((((((((int)blockIdx.z) * 4194304) + (((int)blockIdx.y) * 131072)) + (((int)threadIdx.y) * 16384)) + (ax1_inner * 2048)) + (k_outer * 8)) + ((int)threadIdx.x))];
    }
    #pragma unroll
    for (int ax1_inner_1 = 0; ax1_inner_1 < 8; ++ax1_inner_1) {
      y_shared[(((((int)threadIdx.y) * 64) + (ax1_inner_1 * 8)) + ((int)threadIdx.x))] = y[((((((((int)blockIdx.z) * 4194304) + (((int)blockIdx.x) * 131072)) + (((int)threadIdx.y) * 16384)) + (ax1_inner_1 * 2048)) + (k_outer * 8)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
      #pragma unroll
      for (int ax1 = 0; ax1 < 8; ++ax1) {
        x_shared_local[ax1] = x_shared[(((((int)threadIdx.y) * 64) + (ax1 * 8)) + k_inner)];
      }
      #pragma unroll
      for (int ax1_1 = 0; ax1_1 < 8; ++ax1_1) {
        y_shared_local[ax1_1] = y_shared[(((((int)threadIdx.x) * 64) + (ax1_1 * 8)) + k_inner)];
      }
      for (int i_c = 0; i_c < 8; ++i_c) {
        #pragma unroll
        for (int j_c = 0; j_c < 8; ++j_c) {
          T_batch_matmul_NT_local[((i_c * 8) + j_c)] = (T_batch_matmul_NT_local[((i_c * 8) + j_c)] + (x_shared_local[i_c] * y_shared_local[j_c]));
        }
      }
    }
  }
  for (int i_inner_inner = 0; i_inner_inner < 8; ++i_inner_inner) {
    #pragma unroll
    for (int j_inner_inner = 0; j_inner_inner < 8; ++j_inner_inner) {
      T_batch_matmul_NT[(((((((((int)blockIdx.z) * 4194304) + (((int)blockIdx.y) * 131072)) + (((int)threadIdx.y) * 16384)) + (i_inner_inner * 2048)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + j_inner_inner)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + j_inner_inner)];
    }
  }
}

