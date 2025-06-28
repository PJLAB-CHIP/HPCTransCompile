extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ data, float* __restrict__ depth_to_space) {
  depth_to_space[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = data[((((((((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 2)) % 182) / 7) % 2) * 1456) + (((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 28) % 2) * 728)) + ((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 3)) / 91) * 182)) + ((((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 2)) % 182) / 7) / 2) * 14)) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 28) / 2))];
}

