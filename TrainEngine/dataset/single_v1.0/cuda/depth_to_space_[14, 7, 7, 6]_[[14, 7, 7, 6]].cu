extern "C" __global__ void __launch_bounds__(49) default_function_kernel(float* __restrict__ data, float* __restrict__ depth_to_space) {
  depth_to_space[((((int)blockIdx.x) * 49) + ((int)threadIdx.x))] = data[((((((((((int)blockIdx.x) * 7) + (((int)threadIdx.x) / 7)) / 24) * 294) + ((((((((int)blockIdx.x) * 49) + ((int)threadIdx.x)) % 168) / 12) % 2) * 84)) + ((((((int)blockIdx.x) + ((int)threadIdx.x)) % 12) % 2) * 42)) + ((((((((int)blockIdx.x) * 49) + ((int)threadIdx.x)) % 168) / 12) / 2) * 6)) + (((((int)blockIdx.x) + ((int)threadIdx.x)) % 12) / 2))];
}

