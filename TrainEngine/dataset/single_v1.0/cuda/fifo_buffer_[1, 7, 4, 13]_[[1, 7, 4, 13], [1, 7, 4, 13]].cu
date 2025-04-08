extern "C" __global__ void __launch_bounds__(3) default_function_kernel(float* __restrict__ data, float* __restrict__ new_buffer) {
  if (((((int)blockIdx.x) * 3) + ((int)threadIdx.x)) < 364) {
    new_buffer[((((int)blockIdx.x) * 3) + ((int)threadIdx.x))] = data[((((int)blockIdx.x) * 3) + ((int)threadIdx.x))];
  }
}

