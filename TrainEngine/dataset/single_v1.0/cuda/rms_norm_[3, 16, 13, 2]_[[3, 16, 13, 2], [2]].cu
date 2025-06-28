extern "C" __global__ void __launch_bounds__(16) default_function_kernel_1(float* __restrict__ T_cast, float* __restrict__ T_multiply_red, float* __restrict__ data, float* __restrict__ weight) {
  T_cast[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))] = ((data[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))] * weight[((((((int)blockIdx.x) % 26) * 8) + (((int)threadIdx.x) >> 1)) / 13)]) * (1.000000e+00f / sqrtf(((T_multiply_red[((((((int)blockIdx.x) / 26) * 26) + ((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 1)) % 13) * 2)) + (((int)threadIdx.x) & 1))] * 6.250000e-02f) + 1.000000e-05f))));
}

extern "C" __global__ void default_function_kernel(float* __restrict__ T_multiply_red, float* __restrict__ data) {
  T_multiply_red[((int)blockIdx.x)] = 0.000000e+00f;
  for (int k1 = 0; k1 < 16; ++k1) {
    T_multiply_red[((int)blockIdx.x)] = (T_multiply_red[((int)blockIdx.x)] + (data[((((((int)blockIdx.x) / 26) * 416) + (k1 * 26)) + (((int)blockIdx.x) % 26))] * data[((((((int)blockIdx.x) / 26) * 416) + (k1 * 26)) + (((int)blockIdx.x) % 26))]));
  }
}

