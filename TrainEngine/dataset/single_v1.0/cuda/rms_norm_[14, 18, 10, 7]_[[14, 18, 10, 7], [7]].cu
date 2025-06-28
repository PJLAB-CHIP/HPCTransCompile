extern "C" __global__ void __launch_bounds__(40) default_function_kernel_1(float* __restrict__ T_cast, float* __restrict__ T_multiply_red, float* __restrict__ data, float* __restrict__ weight) {
  T_cast[((((int)blockIdx.x) * 40) + ((int)threadIdx.x))] = ((data[((((int)blockIdx.x) * 40) + ((int)threadIdx.x))] * weight[((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) / 10)) % 126) / 7)]) * (1.000000e+00f / sqrtf(((T_multiply_red[(((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) / 20)) / 63) * 70) + (((((int)blockIdx.x) * 40) + ((int)threadIdx.x)) % 70))] * 5.555556e-02f) + 1.000000e-05f))));
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ T_multiply_red, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 2)) < 245) {
    T_multiply_red[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = 0.000000e+00f;
  }
  for (int k1 = 0; k1 < 18; ++k1) {
    if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 2)) < 245) {
      T_multiply_red[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (T_multiply_red[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + (data[((((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 1)) / 35) * 1260) + (k1 * 70)) + (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) % 70))] * data[((((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 1)) / 35) * 1260) + (k1 * 70)) + (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) % 70))]));
    }
  }
}

