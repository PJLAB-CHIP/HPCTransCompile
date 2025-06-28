extern "C" __global__ void __launch_bounds__(30) default_function_kernel(float* __restrict__ T_strided_slice, float* __restrict__ a) {
  T_strided_slice[((int)threadIdx.x)] = a[(((((((int)threadIdx.x) / 10) * 40) + (((((int)threadIdx.x) % 10) >> 1) * 5)) + (((int)threadIdx.x) & 1)) + 53)];
}

