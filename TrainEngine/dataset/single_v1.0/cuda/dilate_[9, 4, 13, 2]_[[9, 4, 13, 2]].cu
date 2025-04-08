extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ DilatedInput, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 3)) < 117) {
    DilatedInput[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = data[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))];
  }
}

