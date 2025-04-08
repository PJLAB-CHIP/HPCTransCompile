extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ Scale, float* __restrict__ ScaleShift, float* __restrict__ Shift, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 2)) < 11115) {
    ScaleShift[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = ((data[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] * Scale[((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 1)) % 2223) / 171)]) + Shift[((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 1)) % 2223) / 171)]);
  }
}

