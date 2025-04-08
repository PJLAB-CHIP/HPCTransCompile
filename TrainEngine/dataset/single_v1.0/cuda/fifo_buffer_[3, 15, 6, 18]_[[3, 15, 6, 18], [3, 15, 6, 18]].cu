extern "C" __global__ void __launch_bounds__(48) default_function_kernel(float* __restrict__ data, float* __restrict__ new_buffer) {
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) / 12)) < 405) {
    new_buffer[((((int)blockIdx.x) * 48) + ((int)threadIdx.x))] = data[((((int)blockIdx.x) * 48) + ((int)threadIdx.x))];
  }
}

