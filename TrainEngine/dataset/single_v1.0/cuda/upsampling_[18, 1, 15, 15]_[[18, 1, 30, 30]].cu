extern "C" __global__ void __launch_bounds__(60) default_function_kernel(float* __restrict__ data, float* __restrict__ resize) {
  resize[((((int)blockIdx.x) * 60) + ((int)threadIdx.x))] = data[((((((int)blockIdx.x) / 60) * 900) + (((((int)blockIdx.x) % 60) / 2) * 30)) + (((int)threadIdx.x) / 2))];
}

