extern "C" __global__ void __launch_bounds__(16) default_function_kernel(float* __restrict__ T_reverse_sequence, float* __restrict__ data) {
  T_reverse_sequence[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))] = data[(((((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 1)) % 91) * 2) + (((int)threadIdx.x) & 1)) + 1274) - ((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 1)) / 91) * 182))];
}

