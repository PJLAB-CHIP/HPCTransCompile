extern "C" __global__ void __launch_bounds__(35) default_function_kernel_1(float* __restrict__ T_cast, float* __restrict__ T_multiply_red, float* __restrict__ data, float* __restrict__ weight) {
  T_cast[((((int)blockIdx.x) * 35) + ((int)threadIdx.x))] = ((data[((((int)blockIdx.x) * 35) + ((int)threadIdx.x))] * weight[((((((int)blockIdx.x) % 48) * 5) + (((int)threadIdx.x) / 7)) / 24)]) * (1.000000e+00f / sqrtf(((T_multiply_red[(((((int)blockIdx.x) / 48) * 168) + (((((int)blockIdx.x) * 35) + ((int)threadIdx.x)) % 168))] * 1.000000e-01f) + 1.000000e-05f))));
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ T_multiply_red, float* __restrict__ data) {
  T_multiply_red[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = 0.000000e+00f;
  for (int k1 = 0; k1 < 10; ++k1) {
    T_multiply_red[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (T_multiply_red[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + (data[((((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 3)) / 21) * 1680) + (k1 * 168)) + (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) % 168))] * data[((((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 3)) / 21) * 1680) + (k1 * 168)) + (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) % 168))]));
  }
}

