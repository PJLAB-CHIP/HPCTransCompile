extern "C" __global__ void __launch_bounds__(8) default_function_kernel(float* __restrict__ T_reverse_sequence, float* __restrict__ data) {
  T_reverse_sequence[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))] = data[(((((((int)blockIdx.x) % 10) * 8) + ((int)threadIdx.x)) + 1120) - ((((int)blockIdx.x) / 10) * 80))];
}

