extern "C" __global__ void __launch_bounds__(54) default_function_kernel(float* __restrict__ data, float* __restrict__ depth_to_space) {
  depth_to_space[((((int)blockIdx.x) * 54) + ((int)threadIdx.x))] = data[(((((((((int)blockIdx.x) / 26) * 1404) + ((((((((int)blockIdx.x) * 27) + (((int)threadIdx.x) >> 1)) % 234) / 13) % 2) * 702)) + (((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 26) % 2) * 351)) + (((((((int)blockIdx.x) % 26) * 3) + (((int)threadIdx.x) / 18)) / 26) * 117)) + ((((((((int)blockIdx.x) * 27) + (((int)threadIdx.x) >> 1)) % 234) / 13) / 2) * 13)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 26) / 2))];
}

