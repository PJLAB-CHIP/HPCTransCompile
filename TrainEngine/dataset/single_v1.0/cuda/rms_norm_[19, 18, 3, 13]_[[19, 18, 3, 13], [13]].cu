extern "C" __global__ void __launch_bounds__(32) default_function_kernel_1(float* __restrict__ T_cast, float* __restrict__ T_multiply_red, float* __restrict__ data, float* __restrict__ weight) {
  if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 1)) < 6669) {
    T_cast[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = ((data[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] * weight[((((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) % 702) / 39)]) * (1.000000e+00f / sqrtf(((T_multiply_red[(((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 1)) / 351) * 39) + (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) % 39))] * 5.555556e-02f) + 1.000000e-05f))));
  }
}

extern "C" __global__ void __launch_bounds__(20) default_function_kernel(float* __restrict__ T_multiply_red, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 20) + ((int)threadIdx.x)) < 741) {
    T_multiply_red[((((int)blockIdx.x) * 20) + ((int)threadIdx.x))] = 0.000000e+00f;
  }
  for (int k1 = 0; k1 < 18; ++k1) {
    if (((((int)blockIdx.x) * 20) + ((int)threadIdx.x)) < 741) {
      T_multiply_red[((((int)blockIdx.x) * 20) + ((int)threadIdx.x))] = (T_multiply_red[((((int)blockIdx.x) * 20) + ((int)threadIdx.x))] + (data[((((((((int)blockIdx.x) * 20) + ((int)threadIdx.x)) / 39) * 702) + (k1 * 39)) + (((((int)blockIdx.x) * 20) + ((int)threadIdx.x)) % 39))] * data[((((((((int)blockIdx.x) * 20) + ((int)threadIdx.x)) / 39) * 702) + (k1 * 39)) + (((((int)blockIdx.x) * 20) + ((int)threadIdx.x)) % 39))]));
    }
  }
}

