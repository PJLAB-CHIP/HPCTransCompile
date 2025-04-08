extern "C" __global__ void __launch_bounds__(60) default_function_kernel(float* __restrict__ data, float* __restrict__ new_buffer) {
  if (((((int)blockIdx.x) * 10) + (((int)threadIdx.x) / 6)) < 4641) {
    new_buffer[((((int)blockIdx.x) * 60) + ((int)threadIdx.x))] = data[((((int)blockIdx.x) * 60) + ((int)threadIdx.x))];
  }
}

