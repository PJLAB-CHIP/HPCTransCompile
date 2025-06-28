extern "C" __global__ void __launch_bounds__(57) default_function_kernel(float* __restrict__ data, float* __restrict__ resize) {
  resize[((((int)blockIdx.x) * 57) + ((int)threadIdx.x))] = data[((((((((int)blockIdx.x) * 19) + (((int)threadIdx.x) / 3)) / 96) * 72) + ((((((((int)blockIdx.x) * 19) + (((int)threadIdx.x) / 3)) % 96) >> 3) / 2) * 12)) + ((((((int)blockIdx.x) * 9) + ((int)threadIdx.x)) % 24) / 2))];
}

