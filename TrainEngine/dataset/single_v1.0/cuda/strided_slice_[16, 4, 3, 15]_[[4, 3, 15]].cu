extern "C" __global__ void __launch_bounds__(21) default_function_kernel(float* __restrict__ T_strided_slice, float* __restrict__ a) {
  T_strided_slice[((int)threadIdx.x)] = a[((((((int)threadIdx.x) / 7) * 45) + (((int)threadIdx.x) % 7)) + 78)];
}

