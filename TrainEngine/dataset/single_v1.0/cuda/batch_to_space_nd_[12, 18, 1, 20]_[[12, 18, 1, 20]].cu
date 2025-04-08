extern "C" __global__ void __launch_bounds__(20) default_function_kernel(float* __restrict__ T_strided_slice, float* __restrict__ data) {
  T_strided_slice[((((int)blockIdx.x) * 20) + ((int)threadIdx.x))] = data[((((((int)blockIdx.x) & 3) * 1080) + ((((int)blockIdx.x) >> 2) * 20)) + ((int)threadIdx.x))];
}

