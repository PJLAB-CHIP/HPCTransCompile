extern "C" __global__ void __launch_bounds__(16) default_function_kernel(float* __restrict__ data, float* __restrict__ depth_to_space) {
  depth_to_space[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))] = data[((((((((int)blockIdx.x) >> 3) * 152) + (((((int)threadIdx.x) >> 1) % 2) * 64)) + ((((int)threadIdx.x) & 1) * 32)) + ((((int)blockIdx.x) & 7) * 4)) + ((((int)threadIdx.x) >> 1) / 2))];
}

