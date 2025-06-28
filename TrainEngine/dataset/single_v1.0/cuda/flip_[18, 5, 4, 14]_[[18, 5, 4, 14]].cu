extern "C" __global__ void __launch_bounds__(60) default_function_kernel(float* __restrict__ T_reverse_sequence, float* __restrict__ data) {
  T_reverse_sequence[((((int)blockIdx.x) * 60) + ((int)threadIdx.x))] = data[(((((((int)blockIdx.x) * 60) + ((int)threadIdx.x)) % 280) + 4760) - ((((((int)blockIdx.x) * 3) + (((int)threadIdx.x) / 20)) / 14) * 280))];
}

