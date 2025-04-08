extern "C" __global__ void __launch_bounds__(52) default_function_kernel(float* __restrict__ Scale, float* __restrict__ ScaleShift, float* __restrict__ Shift, float* __restrict__ data) {
  ScaleShift[((((int)blockIdx.x) * 52) + ((int)threadIdx.x))] = ((data[((((int)blockIdx.x) * 52) + ((int)threadIdx.x))] * Scale[(((((int)blockIdx.x) / 48) * 13) + (((int)threadIdx.x) % 13))]) + Shift[(((((int)blockIdx.x) / 48) * 13) + (((int)threadIdx.x) % 13))]);
}

