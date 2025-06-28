extern "C" __global__ void __launch_bounds__(12) default_function_kernel(float* __restrict__ data, float* __restrict__ space_to_depth) {
  space_to_depth[((((int)blockIdx.x) * 12) + ((int)threadIdx.x))] = data[((((((((int)blockIdx.x) / 12) * 144) + (((((int)blockIdx.x) % 12) % 3) * 48)) + (((((int)blockIdx.x) % 12) / 6) * 24)) + (((int)threadIdx.x) * 2)) + (((((int)blockIdx.x) % 12) % 6) / 3))];
}

