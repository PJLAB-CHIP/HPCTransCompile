extern "C" __global__ void __launch_bounds__(8) default_function_kernel(float* __restrict__ data, float* __restrict__ space_to_depth) {
  space_to_depth[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))] = data[((((((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 2)) / 9) * 36) + ((((((((int)blockIdx.x) * 8) + ((int)threadIdx.x)) % 36) / 3) % 3) * 12)) + ((((((((int)blockIdx.x) * 8) + ((int)threadIdx.x)) % 36) / 3) / 6) * 6)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 3) * 2)) + ((((((((int)blockIdx.x) * 8) + ((int)threadIdx.x)) % 36) / 3) % 6) / 3))];
}

