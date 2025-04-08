extern "C" __global__ void __launch_bounds__(60) default_function_kernel(float* __restrict__ data, float* __restrict__ resize) {
  resize[((((int)blockIdx.x) * 60) + ((int)threadIdx.x))] = data[((((((((int)blockIdx.x) * 3) + (((int)threadIdx.x) / 20)) / 80) * 400) + ((((((((int)blockIdx.x) * 3) + (((int)threadIdx.x) / 20)) % 80) >> 2) / 2) * 40)) + ((((((int)blockIdx.x) * 60) + ((int)threadIdx.x)) % 80) / 2))];
}

