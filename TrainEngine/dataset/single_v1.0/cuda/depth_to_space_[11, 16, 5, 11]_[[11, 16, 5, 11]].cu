extern "C" __global__ void __launch_bounds__(20) default_function_kernel(float* __restrict__ data, float* __restrict__ depth_to_space) {
  depth_to_space[((((int)blockIdx.x) * 20) + ((int)threadIdx.x))] = data[(((((((((int)blockIdx.x) / 44) * 880) + ((((((((int)blockIdx.x) % 11) * 10) + (((int)threadIdx.x) >> 1)) / 11) % 2) * 440)) + (((((((int)blockIdx.x) * 20) + ((int)threadIdx.x)) % 22) % 2) * 220)) + (((((int)blockIdx.x) % 44) / 11) * 55)) + ((((((((int)blockIdx.x) % 11) * 10) + (((int)threadIdx.x) >> 1)) / 11) / 2) * 11)) + ((((((int)blockIdx.x) * 20) + ((int)threadIdx.x)) % 22) / 2))];
}

