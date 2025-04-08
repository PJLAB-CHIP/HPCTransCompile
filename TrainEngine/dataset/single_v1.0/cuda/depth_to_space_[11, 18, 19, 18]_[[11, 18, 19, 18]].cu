extern "C" __global__ void __launch_bounds__(38) default_function_kernel(float* __restrict__ data, float* __restrict__ depth_to_space) {
  depth_to_space[((((int)blockIdx.x) * 38) + ((int)threadIdx.x))] = data[(((((((((int)blockIdx.x) / 144) * 6156) + ((((((((int)blockIdx.x) % 36) * 19) + (((int)threadIdx.x) >> 1)) / 18) % 2) * 2736)) + (((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 36) % 2) * 1368)) + (((((int)blockIdx.x) % 144) / 36) * 342)) + ((((((((int)blockIdx.x) % 36) * 19) + (((int)threadIdx.x) >> 1)) / 18) / 2) * 18)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 36) / 2))];
}

