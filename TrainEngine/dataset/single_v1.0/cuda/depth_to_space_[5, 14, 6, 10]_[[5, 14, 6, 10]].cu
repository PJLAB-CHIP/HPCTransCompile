extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ data, float* __restrict__ depth_to_space) {
  if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 4)) < 225) {
    depth_to_space[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = data[(((((((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 4)) / 45) * 840) + ((((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 2)) % 60) / 5) % 2) * 360)) + (((((((int)blockIdx.x) * 12) + ((int)threadIdx.x)) % 20) % 2) * 180)) + (((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 4)) % 45) / 15) * 60)) + ((((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 2)) % 60) / 5) / 2) * 10)) + ((((((int)blockIdx.x) * 12) + ((int)threadIdx.x)) % 20) / 2))];
  }
}

