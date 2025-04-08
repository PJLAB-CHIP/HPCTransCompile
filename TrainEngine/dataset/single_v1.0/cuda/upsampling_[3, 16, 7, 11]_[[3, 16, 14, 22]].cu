extern "C" __global__ void __launch_bounds__(44) default_function_kernel(float* __restrict__ data, float* __restrict__ resize) {
  resize[((((int)blockIdx.x) * 44) + ((int)threadIdx.x))] = data[((((((int)blockIdx.x) / 28) * 308) + (((((int)blockIdx.x) % 28) / 2) * 22)) + (((int)threadIdx.x) / 2))];
}

