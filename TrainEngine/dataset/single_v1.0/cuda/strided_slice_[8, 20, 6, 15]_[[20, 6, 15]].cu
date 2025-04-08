extern "C" __global__ void __launch_bounds__(42) default_function_kernel(float* __restrict__ T_strided_slice, float* __restrict__ a) {
  T_strided_slice[((((int)blockIdx.x) * 42) + ((int)threadIdx.x))] = a[(((((((((int)blockIdx.x) * 3) + (((int)threadIdx.x) / 14)) >> 1) * 90) + ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) / 7)) & 3) * 15)) + (((int)threadIdx.x) % 7)) + 123)];
}

