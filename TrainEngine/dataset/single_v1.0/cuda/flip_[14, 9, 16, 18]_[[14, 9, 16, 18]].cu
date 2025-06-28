extern "C" __global__ void __launch_bounds__(48) default_function_kernel(float* __restrict__ T_reverse_sequence, float* __restrict__ data) {
  T_reverse_sequence[((((int)blockIdx.x) * 48) + ((int)threadIdx.x))] = data[(((((((int)blockIdx.x) % 54) * 48) + ((int)threadIdx.x)) + 33696) - ((((int)blockIdx.x) / 54) * 2592))];
}

