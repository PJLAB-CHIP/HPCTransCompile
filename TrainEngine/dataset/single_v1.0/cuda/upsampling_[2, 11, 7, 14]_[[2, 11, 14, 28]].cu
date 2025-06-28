extern "C" __global__ void __launch_bounds__(56) default_function_kernel(float* __restrict__ data, float* __restrict__ resize) {
  resize[((((int)blockIdx.x) * 56) + ((int)threadIdx.x))] = data[((((((int)blockIdx.x) / 28) * 392) + (((((int)blockIdx.x) % 28) / 2) * 28)) + (((int)threadIdx.x) / 2))];
}

