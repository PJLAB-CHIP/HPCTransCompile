extern "C" __global__ void __launch_bounds__(64) default_function_kernel_1(float* __restrict__ T_cast, float* __restrict__ T_multiply_red, float* __restrict__ data, float* __restrict__ weight) {
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 4)) < 9) {
    T_cast[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = ((data[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] * weight[((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 2)) % 18) / 9)]) * (1.000000e+00f / sqrtf(((T_multiply_red[((((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 3)) / 9) * 36) + (((((((int)blockIdx.x) * 28) + ((int)threadIdx.x)) % 36) / 9) * 9)) + ((((int)blockIdx.x) + ((int)threadIdx.x)) % 9))] * 5.000000e-01f) + 1.000000e-05f))));
  }
}

extern "C" __global__ void __launch_bounds__(16) default_function_kernel(float* __restrict__ T_multiply_red, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 3)) < 9) {
    T_multiply_red[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))] = 0.000000e+00f;
  }
  for (int k1 = 0; k1 < 2; ++k1) {
    if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 3)) < 9) {
      T_multiply_red[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))] = (T_multiply_red[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))] + (data[((((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 2)) / 9) * 72) + (k1 * 36)) + (((((int)blockIdx.x) * 16) + ((int)threadIdx.x)) % 36))] * data[((((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 2)) / 9) * 72) + (k1 * 36)) + (((((int)blockIdx.x) * 16) + ((int)threadIdx.x)) % 36))]));
    }
  }
}

