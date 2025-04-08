extern "C" __global__ void __launch_bounds__(45) default_function_kernel(float* __restrict__ T_reverse_sequence, float* __restrict__ data) {
  T_reverse_sequence[((((int)blockIdx.x) * 45) + ((int)threadIdx.x))] = data[(((((((int)blockIdx.x) & 3) * 45) + ((int)threadIdx.x)) + 3240) - ((((int)blockIdx.x) >> 2) * 180))];
}

