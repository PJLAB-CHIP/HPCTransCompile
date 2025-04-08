extern "C" __global__ void default_function_kernel_1(float* __restrict__ T_softmax_expsum, float* __restrict__ T_softmax_maxelem, float* __restrict__ data) {
  T_softmax_expsum[((int)blockIdx.x)] = 0.000000e+00f;
  for (int k = 0; k < 10; ++k) {
    T_softmax_expsum[((int)blockIdx.x)] = (T_softmax_expsum[((int)blockIdx.x)] + __expf((data[((((int)blockIdx.x) * 10) + k)] - T_softmax_maxelem[((int)blockIdx.x)])));
  }
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_2(float* __restrict__ T_softmax_expsum, float* __restrict__ T_softmax_maxelem, float* __restrict__ T_softmax_norm, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 3)) < 75) {
    T_softmax_norm[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (__expf((data[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] - T_softmax_maxelem[(((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 1)) / 5)])) / T_softmax_expsum[(((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 1)) / 5)]);
  }
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ T_softmax_maxelem, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 2)) < 15) {
    T_softmax_maxelem[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = -3.402823e+38f;
  }
  for (int k = 0; k < 10; ++k) {
    if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 2)) < 15) {
      T_softmax_maxelem[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(T_softmax_maxelem[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], data[(((((int)blockIdx.x) * 320) + (((int)threadIdx.x) * 10)) + k)]);
    }
  }
}

