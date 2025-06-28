extern "C" __global__ void __launch_bounds__(28) default_function_kernel(float* __restrict__ Scale, float* __restrict__ ScaleShift, float* __restrict__ Shift, float* __restrict__ data) {
  ScaleShift[((int)threadIdx.x)] = ((data[((int)threadIdx.x)] * Scale[((((int)threadIdx.x) & 3) >> 1)]) + Shift[((((int)threadIdx.x) & 3) >> 1)]);
}

