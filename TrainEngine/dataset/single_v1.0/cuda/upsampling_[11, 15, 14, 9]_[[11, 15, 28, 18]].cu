extern "C" __global__ void __launch_bounds__(54) default_function_kernel(float* __restrict__ data, float* __restrict__ resize) {
  resize[((((int)blockIdx.x) * 54) + ((int)threadIdx.x))] = data[((((((((int)blockIdx.x) * 3) + (((int)threadIdx.x) / 18)) / 112) * 504) + ((((((((int)blockIdx.x) * 3) + (((int)threadIdx.x) / 18)) % 112) >> 1) / 2) * 18)) + ((((((int)blockIdx.x) * 18) + ((int)threadIdx.x)) % 36) / 2))];
}

