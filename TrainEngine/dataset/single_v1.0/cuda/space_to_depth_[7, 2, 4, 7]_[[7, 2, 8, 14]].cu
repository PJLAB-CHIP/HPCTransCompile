extern "C" __global__ void __launch_bounds__(28) default_function_kernel(float* __restrict__ data, float* __restrict__ space_to_depth) {
  space_to_depth[((((int)blockIdx.x) * 28) + ((int)threadIdx.x))] = data[(((((((((int)blockIdx.x) >> 3) * 224) + (((((int)blockIdx.x) & 7) % 2) * 112)) + ((((int)threadIdx.x) / 7) * 28)) + (((((int)blockIdx.x) & 7) / 4) * 14)) + ((((int)threadIdx.x) % 7) * 2)) + (((((int)blockIdx.x) & 7) % 4) / 2))];
}

