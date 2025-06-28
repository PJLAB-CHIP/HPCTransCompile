extern "C" __global__ void __launch_bounds__(26) default_function_kernel(float* __restrict__ T_reverse_sequence, float* __restrict__ data) {
  T_reverse_sequence[((((int)blockIdx.x) * 26) + ((int)threadIdx.x))] = data[(((((((int)blockIdx.x) * 26) + ((int)threadIdx.x)) % 112) + 1344) - ((((((int)blockIdx.x) * 13) + (((int)threadIdx.x) >> 1)) / 56) * 112))];
}

