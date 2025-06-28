extern "C" __global__ void __launch_bounds__(60) default_function_kernel(float* __restrict__ data, float* __restrict__ depth_to_space) {
  depth_to_space[((((int)blockIdx.x) * 60) + ((int)threadIdx.x))] = data[((((((((int)blockIdx.x) >> 3) * 480) + ((((int)threadIdx.x) / 30) * 240)) + (((((int)threadIdx.x) % 30) % 2) * 120)) + ((((int)blockIdx.x) & 7) * 15)) + ((((int)threadIdx.x) % 30) / 2))];
}

