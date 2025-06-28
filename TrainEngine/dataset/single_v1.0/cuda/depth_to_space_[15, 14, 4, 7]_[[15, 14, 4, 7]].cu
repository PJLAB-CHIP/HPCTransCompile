extern "C" __global__ void __launch_bounds__(42) default_function_kernel(float* __restrict__ data, float* __restrict__ depth_to_space) {
  depth_to_space[((((int)blockIdx.x) * 42) + ((int)threadIdx.x))] = data[(((((((((int)blockIdx.x) >> 3) * 392) + (((((((int)blockIdx.x) * 3) + (((int)threadIdx.x) / 14)) & 7) % 2) * 168)) + (((((int)threadIdx.x) % 14) % 2) * 84)) + (((((((int)blockIdx.x) & 7) * 3) + (((int)threadIdx.x) / 14)) >> 3) * 28)) + (((((((int)blockIdx.x) * 3) + (((int)threadIdx.x) / 14)) & 7) / 2) * 7)) + ((((int)threadIdx.x) % 14) / 2))];
}

