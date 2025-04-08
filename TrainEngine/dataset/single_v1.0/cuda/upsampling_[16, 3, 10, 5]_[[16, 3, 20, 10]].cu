extern "C" __global__ void __launch_bounds__(60) default_function_kernel(float* __restrict__ data, float* __restrict__ resize) {
  resize[((((int)blockIdx.x) * 60) + ((int)threadIdx.x))] = data[((((((((int)blockIdx.x) * 3) + (((int)threadIdx.x) / 20)) / 40) * 200) + (((((((int)blockIdx.x) * 3) + (((int)threadIdx.x) / 20)) % 40) / 2) * 10)) + ((((int)threadIdx.x) % 20) / 2))];
}

