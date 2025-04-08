extern "C" __global__ void __launch_bounds__(14) default_function_kernel(float* __restrict__ data, float* __restrict__ depth_to_space) {
  depth_to_space[((((int)blockIdx.x) * 14) + ((int)threadIdx.x))] = data[((((((((((int)blockIdx.x) * 7) + (((int)threadIdx.x) >> 1)) / 102) * 357) + ((((((((int)blockIdx.x) * 7) + (((int)threadIdx.x) >> 1)) % 102) / 17) % 2) * 102)) + (((((((int)blockIdx.x) * 14) + ((int)threadIdx.x)) % 34) % 2) * 51)) + ((((((((int)blockIdx.x) * 7) + (((int)threadIdx.x) >> 1)) % 102) / 17) / 2) * 17)) + ((((((int)blockIdx.x) * 14) + ((int)threadIdx.x)) % 34) / 2))];
}

