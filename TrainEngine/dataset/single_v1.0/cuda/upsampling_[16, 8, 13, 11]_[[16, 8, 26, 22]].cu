extern "C" __global__ void __launch_bounds__(64) default_function_kernel(float* __restrict__ data, float* __restrict__ resize) {
  resize[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = data[((((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 4)) / 143) * 572) + ((((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 2)) % 572) / 11) / 2) * 22)) + ((((((int)blockIdx.x) * 20) + ((int)threadIdx.x)) % 44) / 2))];
}

