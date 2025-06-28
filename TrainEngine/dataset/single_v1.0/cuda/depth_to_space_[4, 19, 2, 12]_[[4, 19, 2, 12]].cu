extern "C" __global__ void __launch_bounds__(16) default_function_kernel(float* __restrict__ data, float* __restrict__ depth_to_space) {
  depth_to_space[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))] = data[(((((((((int)blockIdx.x) / 24) * 456) + ((((((((int)blockIdx.x) % 6) * 2) + (((int)threadIdx.x) >> 3)) / 3) % 2) * 192)) + (((((((int)blockIdx.x) * 16) + ((int)threadIdx.x)) % 24) % 2) * 96)) + (((((int)blockIdx.x) % 24) / 6) * 24)) + ((((((((int)blockIdx.x) % 6) * 2) + (((int)threadIdx.x) >> 3)) / 3) / 2) * 12)) + ((((((int)blockIdx.x) * 16) + ((int)threadIdx.x)) % 24) / 2))];
}

