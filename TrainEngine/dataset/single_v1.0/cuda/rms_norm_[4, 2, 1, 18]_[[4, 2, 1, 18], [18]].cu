extern "C" __global__ void __launch_bounds__(9) default_function_kernel_1(float* __restrict__ T_cast, float* __restrict__ T_multiply_red, float* __restrict__ data, float* __restrict__ weight) {
  T_cast[((((int)blockIdx.x) * 9) + ((int)threadIdx.x))] = ((data[((((int)blockIdx.x) * 9) + ((int)threadIdx.x))] * weight[((((int)blockIdx.x) & 3) >> 1)]) * (1.000000e+00f / sqrtf(((T_multiply_red[((((((int)blockIdx.x) >> 2) * 18) + ((((int)blockIdx.x) & 1) * 9)) + ((int)threadIdx.x))] * 5.000000e-01f) + 1.000000e-05f))));
}

extern "C" __global__ void __launch_bounds__(4) default_function_kernel(float* __restrict__ T_multiply_red, float* __restrict__ data) {
  T_multiply_red[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))] = 0.000000e+00f;
  for (int k1 = 0; k1 < 2; ++k1) {
    T_multiply_red[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))] = (T_multiply_red[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))] + (data[((((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 1)) / 9) * 36) + (k1 * 18)) + (((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 18))] * data[((((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 1)) / 9) * 36) + (k1 * 18)) + (((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 18))]));
  }
}

