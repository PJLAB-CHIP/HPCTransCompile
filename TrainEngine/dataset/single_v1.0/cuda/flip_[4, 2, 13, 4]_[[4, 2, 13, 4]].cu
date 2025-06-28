extern "C" __global__ void __launch_bounds__(13) default_function_kernel(float* __restrict__ T_reverse_sequence, float* __restrict__ data) {
  T_reverse_sequence[((((int)blockIdx.x) * 13) + ((int)threadIdx.x))] = data[(((((((int)blockIdx.x) & 7) * 13) + ((int)threadIdx.x)) + 312) - ((((int)blockIdx.x) >> 3) * 104))];
}

