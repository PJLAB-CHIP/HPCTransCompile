
    Task: Translate the given CUDA code to its equivalent high-performance CPU C code.
    Context: You are provided with a CUDA code snippet that needs to be translated into CPU C code. The translation should preserve the same functionality as much as possible. Focus on translating the CUDA-specific parallel constructs into constructs supported by the CPU, such as using OpenMP for parallelism. The resulting CPU C code should be complete and ready to compile.

    Example 1:
    Input CUDA Code: extern "C" __global__ void __launch_bounds__(18) default_function_kernel(float* __restrict__ compute, float* __restrict__ data) {
  compute[((((int)blockIdx.x) * 18) + ((int)threadIdx.x))] = atanf(data[((((int)blockIdx.x) * 18) + ((int)threadIdx.x))]);
}


    Input Tensor Shape: [[17, 12, 7, 9]]
    Output C Code: void default_function_kernel(float* compute, float* data) {
  #pragma omp parallel for
  for (int32_t i0_i1_fused_i2_fused = 0; i0_i1_fused_i2_fused < 1428; ++i0_i1_fused_i2_fused) {
    for (int32_t i3_s = 0; i3_s < 9; ++i3_s) {
      compute[((i0_i1_fused_i2_fused * 9) + i3_s)] = atanf(data[((i0_i1_fused_i2_fused * 9) + i3_s)]);
    }
  }
}


    //|End-of-Code|

    Example 2:
    Input CUDA Code: extern "C" __global__ void __launch_bounds__(2) default_function_kernel(float* __restrict__ T_matmul, float* __restrict__ left_matrix, float* __restrict__ right_matrix) {
  float T_matmul_local[2];
  __shared__ float left_matrix_shared[4];
  __shared__ float right_matrix_shared[4];
  T_matmul_local[0] = 0.000000e+00f;
  T_matmul_local[1] = 0.000000e+00f;
  for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 2; ++ax0_ax1_fused_outer_outer) {
    left_matrix_shared[((ax0_ax1_fused_outer_outer * 2) + ((int)threadIdx.x))] = left_matrix[((ax0_ax1_fused_outer_outer * 2) + ((int)threadIdx.x))];
  }
  for (int ax0_ax1_fused_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_1 < 2; ++ax0_ax1_fused_outer_outer_1) {
    right_matrix_shared[((ax0_ax1_fused_outer_outer_1 * 2) + ((int)threadIdx.x))] = right_matrix[((ax0_ax1_fused_outer_outer_1 * 2) + ((int)threadIdx.x))];
  }
  __syncthreads();
  for (int k_inner = 0; k_inner < 2; ++k_inner) {
    T_matmul_local[0] = (T_matmul_local[0] + (left_matrix_shared[((((int)threadIdx.x) * 2) + k_inner)] * right_matrix_shared[(k_inner * 2)]));
    T_matmul_local[1] = (T_matmul_local[1] + (left_matrix_shared[((((int)threadIdx.x) * 2) + k_inner)] * right_matrix_shared[((k_inner * 2) + 1)]));
  }
  T_matmul[(((int)threadIdx.x) * 2)] = T_matmul_local[0];
  T_matmul[((((int)threadIdx.x) * 2) + 1)] = T_matmul_local[1];
}


    Input Tensor Shape: [[2, 2], [2, 2]]
    Output C Code: void default_function_kernel(float* T_matmul, float* left_matrix, float* right_matrix) {
  for (int32_t ax1_outer_outer_outer = 0; ax1_outer_outer_outer < 2; ++ax1_outer_outer_outer) {
    for (int32_t ax0_inner_init = 0; ax0_inner_init < 2; ++ax0_inner_init) {
      T_matmul[((ax0_inner_init * 2) + ax1_outer_outer_outer)] = 0.000000e+00f;
    }
    for (int32_t k_inner = 0; k_inner < 2; ++k_inner) {
      for (int32_t ax0_inner = 0; ax0_inner < 2; ++ax0_inner) {
        T_matmul[((ax0_inner * 2) + ax1_outer_outer_outer)] = (T_matmul[((ax0_inner * 2) + ax1_outer_outer_outer)] + (left_matrix[((ax0_inner * 2) + k_inner)] * right_matrix[((k_inner * 2) + ax1_outer_outer_outer)]));
      }
    }
  }
}


    //|End-of-Code|

    Now translate the following CUDA code to its equivalent high-performance CPU C code:
    Input CUDA Code: extern "C" __global__ void __launch_bounds__(16) default_function_kernel(float* __restrict__ conv2d_nchw, float* __restrict__ data, float* __restrict__ kernel) {
  float conv2d_nchw_local[8];
  __shared__ float pad_temp_shared[32];
  __shared__ float kernel_shared[576];
  for (int yy_c_init = 0; yy_c_init < 2; ++yy_c_init) {
    conv2d_nchw_local[yy_c_init] = 0.000000e+00f;
    conv2d_nchw_local[(yy_c_init + 2)] = 0.000000e+00f;
    conv2d_nchw_local[(yy_c_init + 4)] = 0.000000e+00f;
    conv2d_nchw_local[(yy_c_init + 6)] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[(((((int)threadIdx.z) * 4) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] = data[(((((((((((int)blockIdx.z) >> 1) * 2097152) + (rc_outer * 131072)) + ((((int)threadIdx.z) >> 2) * 65536)) + (((int)blockIdx.y) * 512)) + ((((int)threadIdx.z) & 3) * 256)) + (((int)blockIdx.x) * 2)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1 < 36; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1) {
      kernel_shared[(((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 36)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1)] = kernel[(((((((((int)blockIdx.z) & 1) * 9216) + (((int)threadIdx.z) * 1152)) + (((int)threadIdx.x) * 576)) + ((ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1 / 18) * 288)) + (rc_outer * 18)) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1 % 18))];
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 2; ++rc_inner) {
      for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
        for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
          for (int yy_c = 0; yy_c < 2; ++yy_c) {
            conv2d_nchw_local[yy_c] = (conv2d_nchw_local[yy_c] + (pad_temp_shared[(((((rc_inner * 16) + (yy_c * 4)) + (ry_inner * 4)) + ((int)threadIdx.x)) + rx_inner)] * kernel_shared[((((((int)threadIdx.z) * 18) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
            conv2d_nchw_local[(yy_c + 2)] = (conv2d_nchw_local[(yy_c + 2)] + (pad_temp_shared[(((((rc_inner * 16) + (yy_c * 4)) + (ry_inner * 4)) + ((int)threadIdx.x)) + rx_inner)] * kernel_shared[(((((((int)threadIdx.z) * 18) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 144)]));
            conv2d_nchw_local[(yy_c + 4)] = (conv2d_nchw_local[(yy_c + 4)] + (pad_temp_shared[(((((rc_inner * 16) + (yy_c * 4)) + (ry_inner * 4)) + ((int)threadIdx.x)) + rx_inner)] * kernel_shared[(((((((int)threadIdx.z) * 18) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 288)]));
            conv2d_nchw_local[(yy_c + 6)] = (conv2d_nchw_local[(yy_c + 6)] + (pad_temp_shared[(((((rc_inner * 16) + (yy_c * 4)) + (ry_inner * 4)) + ((int)threadIdx.x)) + rx_inner)] * kernel_shared[(((((((int)threadIdx.z) * 18) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 432)]));
          }
        }
      }
    }
  }
  for (int yy_inner_inner_inner = 0; yy_inner_inner_inner < 2; ++yy_inner_inner_inner) {
    conv2d_nchw[((((((((int)blockIdx.z) * 2064512) + (((int)threadIdx.z) * 64516)) + (((int)blockIdx.y) * 508)) + (yy_inner_inner_inner * 254)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x))] = conv2d_nchw_local[yy_inner_inner_inner];
    conv2d_nchw[(((((((((int)blockIdx.z) * 2064512) + (((int)threadIdx.z) * 64516)) + (((int)blockIdx.y) * 508)) + (yy_inner_inner_inner * 254)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 516128)] = conv2d_nchw_local[(yy_inner_inner_inner + 2)];
    conv2d_nchw[(((((((((int)blockIdx.z) * 2064512) + (((int)threadIdx.z) * 64516)) + (((int)blockIdx.y) * 508)) + (yy_inner_inner_inner * 254)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 1032256)] = conv2d_nchw_local[(yy_inner_inner_inner + 4)];
    conv2d_nchw[(((((((((int)blockIdx.z) * 2064512) + (((int)threadIdx.z) * 64516)) + (((int)blockIdx.y) * 508)) + (yy_inner_inner_inner * 254)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)) + 1548384)] = conv2d_nchw_local[(yy_inner_inner_inner + 6)];
  }
}


    Input Tensor Shape: [[2, 32, 256, 256], [64, 32, 3, 3]]
    Print only a single C function implementation, ending with the comment '|End-of-Code|'.
    