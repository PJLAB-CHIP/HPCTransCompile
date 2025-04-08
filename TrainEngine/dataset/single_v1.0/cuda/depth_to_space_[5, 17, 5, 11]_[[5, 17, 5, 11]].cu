extern "C" __global__ void __launch_bounds__(50) default_function_kernel(float* __restrict__ data, float* __restrict__ depth_to_space) {
  depth_to_space[((((int)blockIdx.x) * 50) + ((int)threadIdx.x))] = data[(((((((((((int)blockIdx.x) * 5) + (((int)threadIdx.x) / 10)) / 88) * 935) + ((((((((int)blockIdx.x) * 25) + (((int)threadIdx.x) >> 1)) % 110) / 11) % 2) * 440)) + (((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) % 22) % 2) * 220)) + (((((((int)blockIdx.x) * 5) + (((int)threadIdx.x) / 10)) % 88) / 22) * 55)) + ((((((((int)blockIdx.x) * 25) + (((int)threadIdx.x) >> 1)) % 110) / 11) / 2) * 11)) + ((((((int)blockIdx.x) * 6) + ((int)threadIdx.x)) % 22) / 2))];
}

