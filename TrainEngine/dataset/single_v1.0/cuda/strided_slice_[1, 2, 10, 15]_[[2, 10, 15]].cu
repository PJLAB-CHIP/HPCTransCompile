extern "C" __global__ void __launch_bounds__(7) default_function_kernel(float* __restrict__ T_strided_slice, float* __restrict__ a) {
  T_strided_slice[((((int)blockIdx.x) * 7) + ((int)threadIdx.x))] = a[(((((int)blockIdx.x) * 15) + ((int)threadIdx.x)) + 183)];
}

