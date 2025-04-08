extern "C" __global__ void __launch_bounds__(40) default_function_kernel(float* __restrict__ T_reverse_sequence, float* __restrict__ data) {
  T_reverse_sequence[((((int)blockIdx.x) * 40) + ((int)threadIdx.x))] = data[(((((((int)blockIdx.x) * 40) + ((int)threadIdx.x)) % 224) + 4256) - ((((((int)blockIdx.x) * 5) + (((int)threadIdx.x) >> 3)) / 28) * 224))];
}

