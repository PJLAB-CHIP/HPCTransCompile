extern "C" __global__ void __launch_bounds__(64) default_function_kernel(float* __restrict__ data, float* __restrict__ depth_to_space) {
  depth_to_space[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = data[(((((((((int)blockIdx.x) / 12) * 768) + ((((((((int)blockIdx.x) % 3) * 32) + (((int)threadIdx.x) >> 1)) / 3) % 2) * 384)) + (((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 6) % 2) * 192)) + (((((int)blockIdx.x) % 12) / 3) * 48)) + ((((((((int)blockIdx.x) % 3) * 32) + (((int)threadIdx.x) >> 1)) / 3) / 2) * 3)) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 6) / 2))];
}

