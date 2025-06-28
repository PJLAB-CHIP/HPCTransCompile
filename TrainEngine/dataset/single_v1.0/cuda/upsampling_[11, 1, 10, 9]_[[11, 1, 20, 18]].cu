extern "C" __global__ void __launch_bounds__(36) default_function_kernel(float* __restrict__ data, float* __restrict__ resize) {
  resize[((((int)blockIdx.x) * 36) + ((int)threadIdx.x))] = data[((((((int)blockIdx.x) / 40) * 360) + (((((int)blockIdx.x) % 40) / 2) * 18)) + (((int)threadIdx.x) / 2))];
}

