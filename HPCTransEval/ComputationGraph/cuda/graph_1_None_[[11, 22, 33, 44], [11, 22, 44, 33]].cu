extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_4(float* __restrict__ T_transpose, float* __restrict__ T_transpose_1) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 32768) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) >> 3)) < 43923) {
      T_transpose[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = T_transpose_1[(((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 32768) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) >> 3)) % 1331) / 121) * 31944) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 32768) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) >> 3)) / 1331) * 968)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 44) * 22)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) >> 2)) % 242) / 11))];
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_1(float* __restrict__ T_transpose, float* __restrict__ ph) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 32768) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) >> 3)) < 43923) {
      T_transpose[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = ph[((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 11) * 31944) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 7986) / 363) * 1452)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 131072) + (((int)blockIdx.x) * 512)) + (((int)threadIdx.x) >> 1)) / 3993) * 33)) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 363) / 11))];
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_2(float* __restrict__ T_transpose, float* __restrict__ T_transpose_1) {
  for (int ax0_ax1_fused_outer = 0; ax0_ax1_fused_outer < 2; ++ax0_ax1_fused_outer) {
    if ((((ax0_ax1_fused_outer * 32768) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) >> 3)) < 43923) {
      T_transpose[(((ax0_ax1_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = T_transpose_1[((((((ax0_ax1_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 968) * 363) + ((((ax0_ax1_fused_outer * 32768) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) >> 3)) / 121))];
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ T_transpose, float* __restrict__ ph) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 32768) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) >> 3)) < 43923) {
      T_transpose[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = ph[((((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 32768) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) >> 3)) / 3993) * 31944) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.x) >> 2)) % 242) / 11) * 1452)) + ((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 32768) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) >> 3)) % 3993) / 121) * 44)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 44))];
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_3(float* __restrict__ T_transpose, float* __restrict__ T_transpose_1) {
  for (int ax0_ax1_fused_outer = 0; ax0_ax1_fused_outer < 2; ++ax0_ax1_fused_outer) {
    if ((((ax0_ax1_fused_outer * 32768) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.x) >> 3)) < 43923) {
      T_transpose[(((ax0_ax1_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = (T_transpose[(((ax0_ax1_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] + T_transpose_1[(((ax0_ax1_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))]);
    }
  }
}

