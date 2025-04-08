extern "C" __global__ void __launch_bounds__(54) default_function_kernel(float* __restrict__ data, float* __restrict__ depth_to_space) {
  depth_to_space[((((int)blockIdx.x) * 54) + ((int)threadIdx.x))] = data[((((((((int)blockIdx.x) >> 2) * 270) + (((((((int)blockIdx.x) & 3) * 9) + (((int)threadIdx.x) / 6)) % 2) * 108)) + (((((int)threadIdx.x) % 6) % 2) * 54)) + (((((((int)blockIdx.x) & 3) * 9) + (((int)threadIdx.x) / 6)) / 2) * 3)) + ((((int)threadIdx.x) % 6) / 2))];
}

