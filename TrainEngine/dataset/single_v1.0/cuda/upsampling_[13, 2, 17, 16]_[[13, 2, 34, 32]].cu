extern "C" __global__ void __launch_bounds__(64) default_function_kernel(float* __restrict__ data, float* __restrict__ resize) {
  resize[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = data[((((((int)blockIdx.x) / 68) * 1088) + (((((int)blockIdx.x) % 68) / 2) * 32)) + (((int)threadIdx.x) / 2))];
}

