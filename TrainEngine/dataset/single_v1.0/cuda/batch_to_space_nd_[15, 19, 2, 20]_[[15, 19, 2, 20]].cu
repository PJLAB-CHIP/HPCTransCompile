extern "C" __global__ void __launch_bounds__(24) default_function_kernel(float* __restrict__ T_strided_slice, float* __restrict__ data) {
  T_strided_slice[((((int)blockIdx.x) * 24) + ((int)threadIdx.x))] = data[(((((((((((int)blockIdx.x) * 3) + (((int)threadIdx.x) >> 3)) % 20) / 10) * 4560) + (((((((int)blockIdx.x) * 6) + (((int)threadIdx.x) >> 2)) % 10) / 5) * 2280)) + ((((((int)blockIdx.x) * 3) + (((int)threadIdx.x) >> 3)) / 20) * 40)) + (((((((int)blockIdx.x) * 3) + (((int)threadIdx.x) >> 3)) % 10) / 5) * 20)) + (((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 20))];
}

