extern "C" __global__ void __launch_bounds__(48) default_function_kernel(float* __restrict__ data, float* __restrict__ depth_to_space) {
  depth_to_space[((((int)blockIdx.x) * 48) + ((int)threadIdx.x))] = data[(((((((((int)blockIdx.x) / 12) * 576) + (((((int)threadIdx.x) / 12) % 2) * 288)) + (((((int)threadIdx.x) % 12) % 2) * 144)) + ((((int)blockIdx.x) % 12) * 12)) + (((((int)threadIdx.x) / 12) / 2) * 6)) + ((((int)threadIdx.x) % 12) / 2))];
}

