extern "C" __global__ void __launch_bounds__(64) default_function_kernel_1(float* __restrict__ T_cast, float* __restrict__ T_multiply_red, float* __restrict__ data, float* __restrict__ weight) {
  T_cast[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = ((data[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] * weight[((((int)blockIdx.x) % 25) / 5)]) * (1.000000e+00f / sqrtf(((T_multiply_red[((((((int)blockIdx.x) / 25) * 320) + ((((int)blockIdx.x) % 5) * 64)) + ((int)threadIdx.x))] * 2.000000e-01f) + 1.000000e-05f))));
}

extern "C" __global__ void __launch_bounds__(64) default_function_kernel(float* __restrict__ T_multiply_red, float* __restrict__ data) {
  T_multiply_red[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = 0.000000e+00f;
  for (int k1 = 0; k1 < 5; ++k1) {
    T_multiply_red[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = (T_multiply_red[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] + (data[(((((((int)blockIdx.x) / 5) * 1600) + (k1 * 320)) + ((((int)blockIdx.x) % 5) * 64)) + ((int)threadIdx.x))] * data[(((((((int)blockIdx.x) / 5) * 1600) + (k1 * 320)) + ((((int)blockIdx.x) % 5) * 64)) + ((int)threadIdx.x))]));
  }
}

