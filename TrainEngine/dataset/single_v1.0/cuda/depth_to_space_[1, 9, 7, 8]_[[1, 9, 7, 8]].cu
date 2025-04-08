extern "C" __global__ void __launch_bounds__(8) default_function_kernel(float* __restrict__ data, float* __restrict__ depth_to_space) {
  depth_to_space[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))] = data[(((((((((((int)blockIdx.x) % 28) >> 1) % 2) * 224) + ((((int)threadIdx.x) % 2) * 112)) + ((((int)blockIdx.x) / 28) * 56)) + ((((((int)blockIdx.x) % 28) >> 1) / 2) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + (((int)threadIdx.x) / 2))];
}

