extern "C" __global__ void __launch_bounds__(21) default_function_kernel(float* __restrict__ T_strided_slice, float* __restrict__ a) {
  T_strided_slice[((((int)blockIdx.x) * 21) + ((int)threadIdx.x))] = a[(((((((((int)blockIdx.x) * 3) + (((int)threadIdx.x) / 7)) >> 2) * 102) + ((((((int)blockIdx.x) * 3) + (((int)threadIdx.x) / 7)) & 3) * 17)) + (((int)threadIdx.x) % 7)) + 139)];
}

