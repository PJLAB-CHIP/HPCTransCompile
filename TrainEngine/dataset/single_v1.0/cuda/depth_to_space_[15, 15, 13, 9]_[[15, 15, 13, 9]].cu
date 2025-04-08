extern "C" __global__ void __launch_bounds__(60) default_function_kernel(float* __restrict__ data, float* __restrict__ depth_to_space) {
  depth_to_space[((((int)blockIdx.x) * 60) + ((int)threadIdx.x))] = data[(((((((((((int)blockIdx.x) * 5) + (((int)threadIdx.x) / 12)) / 117) * 1755) + ((((((((int)blockIdx.x) * 10) + (((int)threadIdx.x) / 6)) % 78) / 3) % 2) * 702)) + (((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) % 18) % 2) * 351)) + (((((((int)blockIdx.x) * 5) + (((int)threadIdx.x) / 12)) % 117) / 39) * 117)) + ((((((((int)blockIdx.x) * 10) + (((int)threadIdx.x) / 6)) % 78) / 3) / 2) * 9)) + ((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) % 18) / 2))];
}

