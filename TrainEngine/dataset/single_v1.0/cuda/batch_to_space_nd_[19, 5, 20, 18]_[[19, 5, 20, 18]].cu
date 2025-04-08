extern "C" __global__ void __launch_bounds__(60) default_function_kernel(float* __restrict__ T_strided_slice, float* __restrict__ data) {
  T_strided_slice[((((int)blockIdx.x) * 60) + ((int)threadIdx.x))] = data[(((((((((int)blockIdx.x) % 24) / 12) * 14400) + (((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) / 6)) % 6) / 3) * 7200)) + ((((int)blockIdx.x) / 24) * 360)) + (((((((int)blockIdx.x) % 12) * 5) + (((int)threadIdx.x) / 12)) / 3) * 18)) + (((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) % 18))];
}

