extern "C" __global__ void __launch_bounds__(24) default_function_kernel(float* __restrict__ data, float* __restrict__ depth_to_space) {
  depth_to_space[((((int)blockIdx.x) * 24) + ((int)threadIdx.x))] = data[(((((((((((int)blockIdx.x) * 3) + (((int)threadIdx.x) >> 3)) / 40) * 360) + (((((((int)blockIdx.x) * 3) + (((int)threadIdx.x) >> 3)) % 20) % 2) * 160)) + (((((int)threadIdx.x) & 7) % 2) * 80)) + (((((((int)blockIdx.x) * 3) + (((int)threadIdx.x) >> 3)) % 40) / 20) * 40)) + (((((((int)blockIdx.x) * 3) + (((int)threadIdx.x) >> 3)) % 20) / 2) * 4)) + ((((int)threadIdx.x) & 7) / 2))];
}

