extern "C" __global__ void __launch_bounds__(512) default_function_kernel(float* __restrict__ T_matmul_NN, float* __restrict__ ph, float* __restrict__ ph_1) {
  T_matmul_NN[((int)threadIdx.x)] = 0.000000e+00f;
  for (int k = 0; k < 1000; ++k) {
    T_matmul_NN[((int)threadIdx.x)] = (T_matmul_NN[((int)threadIdx.x)] + (ph[(((((int)threadIdx.x) >> 8) * 1000) + k)] * ph_1[((k * 256) + (((int)threadIdx.x) & 255))]));
  }
}

extern "C" __global__ void __launch_bounds__(256) default_function_kernel_21(float* __restrict__ T_add, float* __restrict__ T_matmul_NN, float* __restrict__ ph) {
  T_add[((int)threadIdx.x)] = (T_matmul_NN[((int)threadIdx.x)] + ph[(((int)threadIdx.x) & 127)]);
}

extern "C" __global__ void __launch_bounds__(512) default_function_kernel_2(float* __restrict__ T_matmul_NN, float* __restrict__ T_matmul_NN_1) {
  T_matmul_NN[((int)threadIdx.x)] = (T_matmul_NN[((int)threadIdx.x)] + T_matmul_NN_1[((int)threadIdx.x)]);
}

extern "C" __global__ void __launch_bounds__(512) default_function_kernel_19(float* __restrict__ T_matmul_NN, float* __restrict__ T_matmul_NN_1) {
  T_matmul_NN[((int)threadIdx.x)] = (T_matmul_NN[((int)threadIdx.x)] + T_matmul_NN_1[((int)threadIdx.x)]);
}

extern "C" __global__ void __launch_bounds__(512) default_function_kernel_5(float* __restrict__ T_matmul_NN, float* __restrict__ T_matmul_NN_1, float* __restrict__ ph) {
  T_matmul_NN[((int)threadIdx.x)] = (T_matmul_NN_1[((int)threadIdx.x)] * ph[((int)threadIdx.x)]);
}

extern "C" __global__ void __launch_bounds__(512) default_function_kernel_17(float* __restrict__ T_matmul_NN) {
  T_matmul_NN[((int)threadIdx.x)] = tanhf(T_matmul_NN[((int)threadIdx.x)]);
}

extern "C" __global__ void __launch_bounds__(512) default_function_kernel_14(float* __restrict__ T_matmul_NN, float* __restrict__ T_matmul_NN_1, float* __restrict__ ph) {
  T_matmul_NN[((int)threadIdx.x)] = 0.000000e+00f;
  for (int k = 0; k < 256; ++k) {
    T_matmul_NN[((int)threadIdx.x)] = (T_matmul_NN[((int)threadIdx.x)] + (T_matmul_NN_1[(((((int)threadIdx.x) >> 8) * 256) + k)] * ph[((k * 256) + (((int)threadIdx.x) & 255))]));
  }
}

extern "C" __global__ void __launch_bounds__(512) default_function_kernel_16(float* __restrict__ T_matmul_NN, float* __restrict__ ph) {
  T_matmul_NN[((int)threadIdx.x)] = (T_matmul_NN[((int)threadIdx.x)] + ph[(((int)threadIdx.x) & 255)]);
}

extern "C" __global__ void __launch_bounds__(512) default_function_kernel_4(float* __restrict__ T_matmul_NN) {
  T_matmul_NN[((int)threadIdx.x)] = (1.000000e+00f / (1.000000e+00f + __expf((0.000000e+00f - T_matmul_NN[((int)threadIdx.x)]))));
}

extern "C" __global__ void __launch_bounds__(512) default_function_kernel_8(float* __restrict__ T_matmul_NN, float* __restrict__ ph, float* __restrict__ ph_1) {
  T_matmul_NN[((int)threadIdx.x)] = 0.000000e+00f;
  for (int k = 0; k < 1000; ++k) {
    T_matmul_NN[((int)threadIdx.x)] = (T_matmul_NN[((int)threadIdx.x)] + (ph[(((((int)threadIdx.x) >> 8) * 1000) + k)] * ph_1[((k * 256) + (((int)threadIdx.x) & 255))]));
  }
}

extern "C" __global__ void __launch_bounds__(512) default_function_kernel_13(float* __restrict__ T_matmul_NN, float* __restrict__ ph) {
  T_matmul_NN[((int)threadIdx.x)] = (T_matmul_NN[((int)threadIdx.x)] * ph[((int)threadIdx.x)]);
}

extern "C" __global__ void __launch_bounds__(512) default_function_kernel_3(float* __restrict__ T_matmul_NN, float* __restrict__ ph) {
  T_matmul_NN[((int)threadIdx.x)] = (T_matmul_NN[((int)threadIdx.x)] + ph[(((int)threadIdx.x) & 255)]);
}

extern "C" __global__ void __launch_bounds__(512) default_function_kernel_12(float* __restrict__ T_matmul_NN) {
  T_matmul_NN[((int)threadIdx.x)] = (1.000000e+00f / (1.000000e+00f + __expf((0.000000e+00f - T_matmul_NN[((int)threadIdx.x)]))));
}

extern "C" __global__ void __launch_bounds__(512) default_function_kernel_11(float* __restrict__ T_matmul_NN, float* __restrict__ ph) {
  T_matmul_NN[((int)threadIdx.x)] = (T_matmul_NN[((int)threadIdx.x)] + ph[(((int)threadIdx.x) & 255)]);
}

extern "C" __global__ void __launch_bounds__(512) default_function_kernel_9(float* __restrict__ T_matmul_NN, float* __restrict__ ph, float* __restrict__ ph_1) {
  T_matmul_NN[((int)threadIdx.x)] = 0.000000e+00f;
  for (int k = 0; k < 256; ++k) {
    T_matmul_NN[((int)threadIdx.x)] = (T_matmul_NN[((int)threadIdx.x)] + (ph[(((((int)threadIdx.x) >> 8) * 256) + k)] * ph_1[((k * 256) + (((int)threadIdx.x) & 255))]));
  }
}

extern "C" __global__ void __launch_bounds__(512) default_function_kernel_10(float* __restrict__ T_matmul_NN, float* __restrict__ T_matmul_NN_1) {
  T_matmul_NN[((int)threadIdx.x)] = (T_matmul_NN[((int)threadIdx.x)] + T_matmul_NN_1[((int)threadIdx.x)]);
}

extern "C" __global__ void __launch_bounds__(512) default_function_kernel_7(float* __restrict__ T_matmul_NN, float* __restrict__ ph, float* __restrict__ ph_1) {
  T_matmul_NN[((int)threadIdx.x)] = 0.000000e+00f;
  for (int k = 0; k < 1000; ++k) {
    T_matmul_NN[((int)threadIdx.x)] = (T_matmul_NN[((int)threadIdx.x)] + (ph[(((((int)threadIdx.x) >> 8) * 1000) + k)] * ph_1[((k * 256) + (((int)threadIdx.x) & 255))]));
  }
}

extern "C" __global__ void __launch_bounds__(512) default_function_kernel_15(float* __restrict__ T_matmul_NN, float* __restrict__ T_matmul_NN_1) {
  T_matmul_NN[((int)threadIdx.x)] = (T_matmul_NN[((int)threadIdx.x)] + T_matmul_NN_1[((int)threadIdx.x)]);
}

extern "C" __global__ void __launch_bounds__(512) default_function_kernel_18(float* __restrict__ T_matmul_NN, float* __restrict__ T_matmul_NN_1) {
  T_matmul_NN[((int)threadIdx.x)] = (T_matmul_NN[((int)threadIdx.x)] * T_matmul_NN_1[((int)threadIdx.x)]);
}

extern "C" __global__ void __launch_bounds__(256) default_function_kernel_20(float* __restrict__ T_matmul_NN, float* __restrict__ T_matmul_NN_1, float* __restrict__ ph) {
  T_matmul_NN[((int)threadIdx.x)] = 0.000000e+00f;
  for (int k = 0; k < 256; ++k) {
    T_matmul_NN[((int)threadIdx.x)] = (T_matmul_NN[((int)threadIdx.x)] + (T_matmul_NN_1[(((((int)threadIdx.x) >> 7) * 256) + k)] * ph[((k * 128) + (((int)threadIdx.x) & 127))]));
  }
}

extern "C" __global__ void __launch_bounds__(512) default_function_kernel_6(float* __restrict__ T_matmul_NN) {
  T_matmul_NN[((int)threadIdx.x)] = (1.000000e+00f - T_matmul_NN[((int)threadIdx.x)]);
}

extern "C" __global__ void __launch_bounds__(512) default_function_kernel_1(float* __restrict__ T_matmul_NN, float* __restrict__ ph, float* __restrict__ ph_1) {
  T_matmul_NN[((int)threadIdx.x)] = 0.000000e+00f;
  for (int k = 0; k < 256; ++k) {
    T_matmul_NN[((int)threadIdx.x)] = (T_matmul_NN[((int)threadIdx.x)] + (ph[(((((int)threadIdx.x) >> 8) * 256) + k)] * ph_1[((k * 256) + (((int)threadIdx.x) & 255))]));
  }
}

