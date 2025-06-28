extern "C" __global__ void __launch_bounds__(50) default_function_kernel(float* __restrict__ T_reverse_sequence, float* __restrict__ data) {
  T_reverse_sequence[((((int)blockIdx.x) * 50) + ((int)threadIdx.x))] = data[(((((((int)blockIdx.x) % 38) * 50) + ((int)threadIdx.x)) + 5700) - ((((int)blockIdx.x) / 38) * 1900))];
}

