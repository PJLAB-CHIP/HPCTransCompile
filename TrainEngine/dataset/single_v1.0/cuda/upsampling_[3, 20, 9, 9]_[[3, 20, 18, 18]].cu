extern "C" __global__ void __launch_bounds__(48) default_function_kernel(float* __restrict__ data, float* __restrict__ resize) {
  resize[((((int)blockIdx.x) * 48) + ((int)threadIdx.x))] = data[((((((int)blockIdx.x) / 27) * 324) + ((((((((int)blockIdx.x) % 27) * 4) + (((int)threadIdx.x) / 12)) / 3) / 2) * 18)) + ((((((int)blockIdx.x) * 12) + ((int)threadIdx.x)) % 36) / 2))];
}

