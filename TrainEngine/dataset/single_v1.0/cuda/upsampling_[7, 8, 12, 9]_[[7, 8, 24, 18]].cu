extern "C" __global__ void __launch_bounds__(64) default_function_kernel(float* __restrict__ data, float* __restrict__ resize) {
  resize[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = data[((((((int)blockIdx.x) / 27) * 432) + ((((((((int)blockIdx.x) % 27) * 16) + (((int)threadIdx.x) >> 2)) / 9) / 2) * 18)) + ((((((int)blockIdx.x) * 28) + ((int)threadIdx.x)) % 36) / 2))];
}

