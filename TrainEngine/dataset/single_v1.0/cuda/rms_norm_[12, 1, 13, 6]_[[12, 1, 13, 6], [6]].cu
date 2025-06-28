extern "C" __global__ void __launch_bounds__(24) default_function_kernel_1(float* __restrict__ T_cast, float* __restrict__ T_multiply_red, float* __restrict__ data, float* __restrict__ weight) {
  T_cast[((((int)blockIdx.x) * 24) + ((int)threadIdx.x))] = ((data[((((int)blockIdx.x) * 24) + ((int)threadIdx.x))] * weight[0]) * (1.000000e+00f / sqrtf((T_multiply_red[((((int)blockIdx.x) * 24) + ((int)threadIdx.x))] + 1.000000e-05f))));
}

extern "C" __global__ void __launch_bounds__(16) default_function_kernel(float* __restrict__ T_multiply_red, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 3)) < 117) {
    T_multiply_red[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))] = 0.000000e+00f;
  }
  if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 3)) < 117) {
    T_multiply_red[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))] = (T_multiply_red[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))] + (data[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))] * data[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))]));
  }
}

