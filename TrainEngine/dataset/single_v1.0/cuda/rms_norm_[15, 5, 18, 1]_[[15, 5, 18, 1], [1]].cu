extern "C" __global__ void __launch_bounds__(28) default_function_kernel_1(float* __restrict__ T_cast, float* __restrict__ T_multiply_red, float* __restrict__ data, float* __restrict__ weight) {
  if (((((int)blockIdx.x) * 14) + (((int)threadIdx.x) >> 1)) < 675) {
    T_cast[((((int)blockIdx.x) * 28) + ((int)threadIdx.x))] = ((data[((((int)blockIdx.x) * 28) + ((int)threadIdx.x))] * weight[((((((int)blockIdx.x) * 14) + (((int)threadIdx.x) >> 1)) % 45) / 9)]) * (1.000000e+00f / sqrtf(((T_multiply_red[(((((((int)blockIdx.x) * 14) + (((int)threadIdx.x) >> 1)) / 45) * 18) + (((((int)blockIdx.x) * 10) + ((int)threadIdx.x)) % 18))] * 2.000000e-01f) + 1.000000e-05f))));
  }
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ T_multiply_red, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 1)) < 135) {
    T_multiply_red[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = 0.000000e+00f;
  }
  for (int k1 = 0; k1 < 5; ++k1) {
    if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 1)) < 135) {
      T_multiply_red[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (T_multiply_red[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + (data[((((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 1)) / 9) * 90) + (k1 * 18)) + (((((int)blockIdx.x) * 14) + ((int)threadIdx.x)) % 18))] * data[((((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 1)) / 9) * 90) + (k1 * 18)) + (((((int)blockIdx.x) * 14) + ((int)threadIdx.x)) % 18))]));
    }
  }
}

