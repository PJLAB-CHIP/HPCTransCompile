extern "C" __global__ void __launch_bounds__(35) default_function_kernel(float* __restrict__ T_strided_slice, float* __restrict__ a) {
  T_strided_slice[((((int)blockIdx.x) * 35) + ((int)threadIdx.x))] = a[((((((int)blockIdx.x) * 200) + ((((int)threadIdx.x) / 7) * 20)) + (((int)threadIdx.x) % 7)) + 243)];
}

