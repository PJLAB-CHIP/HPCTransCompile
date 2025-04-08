extern "C" __global__ void __launch_bounds__(48) default_function_kernel(float* __restrict__ Scale, float* __restrict__ ScaleShift, float* __restrict__ Shift, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 6) + (((int)threadIdx.x) >> 3)) < 3325) {
    ScaleShift[((((int)blockIdx.x) * 48) + ((int)threadIdx.x))] = ((data[((((int)blockIdx.x) * 48) + ((int)threadIdx.x))] * Scale[((((((int)blockIdx.x) * 24) + (((int)threadIdx.x) >> 1)) % 1330) / 133)]) + Shift[((((((int)blockIdx.x) * 24) + (((int)threadIdx.x) >> 1)) % 1330) / 133)]);
  }
}

