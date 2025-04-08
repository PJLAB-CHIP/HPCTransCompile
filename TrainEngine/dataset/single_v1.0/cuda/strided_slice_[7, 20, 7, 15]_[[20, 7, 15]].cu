extern "C" __global__ void default_function_kernel(float* __restrict__ T_strided_slice, float* __restrict__ a) {
  T_strided_slice[((int)blockIdx.x)] = a[(((((((int)blockIdx.x) / 35) * 105) + (((((int)blockIdx.x) % 35) / 7) * 15)) + (((int)blockIdx.x) % 7)) + 138)];
}

