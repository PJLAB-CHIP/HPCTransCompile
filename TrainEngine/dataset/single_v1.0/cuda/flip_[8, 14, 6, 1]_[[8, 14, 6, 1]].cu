extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ T_reverse_sequence, float* __restrict__ data) {
  T_reverse_sequence[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = data[(((((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) % 84) + 588) - ((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 2)) / 21) * 84))];
}

