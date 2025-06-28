extern "C" __global__ void __launch_bounds__(50) default_function_kernel(float* __restrict__ data, float* __restrict__ resize) {
  resize[((((int)blockIdx.x) * 50) + ((int)threadIdx.x))] = data[((((((int)blockIdx.x) / 24) * 300) + ((((((((int)blockIdx.x) % 24) * 5) + (((int)threadIdx.x) / 10)) / 6) / 2) * 30)) + ((((((int)blockIdx.x) * 50) + ((int)threadIdx.x)) % 60) / 2))];
}

