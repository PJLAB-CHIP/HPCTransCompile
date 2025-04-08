extern "C" __global__ void __launch_bounds__(8) default_function_kernel_1(float* __restrict__ T_cast, float* __restrict__ T_multiply_red, float* __restrict__ data, float* __restrict__ weight) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.x)) < 5355) {
    T_cast[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))] = ((data[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))] * weight[((((((int)blockIdx.x) * 8) + ((int)threadIdx.x)) % 765) / 255)]) * (1.000000e+00f / sqrtf(((T_multiply_red[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.x)) / 765) * 255) + (((((int)blockIdx.x) * 8) + ((int)threadIdx.x)) % 255))] * 3.333333e-01f) + 1.000000e-05f))));
  }
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ T_multiply_red, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) < 1785) {
    T_multiply_red[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = 0.000000e+00f;
  }
  for (int k1 = 0; k1 < 3; ++k1) {
    if (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) < 1785) {
      T_multiply_red[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (T_multiply_red[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + (data[((((((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) / 255) * 765) + (k1 * 255)) + (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) % 255))] * data[((((((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) / 255) * 765) + (k1 * 255)) + (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) % 255))]));
    }
  }
}

