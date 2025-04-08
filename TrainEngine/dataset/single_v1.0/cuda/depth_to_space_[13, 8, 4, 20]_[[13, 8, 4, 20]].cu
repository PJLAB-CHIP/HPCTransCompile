extern "C" __global__ void __launch_bounds__(52) default_function_kernel(float* __restrict__ data, float* __restrict__ depth_to_space) {
  depth_to_space[((((int)blockIdx.x) * 52) + ((int)threadIdx.x))] = data[(((((((((((int)blockIdx.x) * 13) + (((int)threadIdx.x) >> 2)) / 160) * 640) + ((((((((int)blockIdx.x) * 13) + (((int)threadIdx.x) >> 2)) % 80) / 10) % 2) * 320)) + (((((((int)blockIdx.x) * 12) + ((int)threadIdx.x)) % 40) % 2) * 160)) + (((((((int)blockIdx.x) * 13) + (((int)threadIdx.x) >> 2)) % 160) / 80) * 80)) + ((((((((int)blockIdx.x) * 13) + (((int)threadIdx.x) >> 2)) % 80) / 10) / 2) * 20)) + ((((((int)blockIdx.x) * 12) + ((int)threadIdx.x)) % 40) / 2))];
}

