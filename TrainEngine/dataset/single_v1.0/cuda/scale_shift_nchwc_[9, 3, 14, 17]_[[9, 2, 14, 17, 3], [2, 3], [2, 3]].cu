extern "C" __global__ void __launch_bounds__(28) default_function_kernel(float* __restrict__ Scale, float* __restrict__ ScaleShift, float* __restrict__ Shift, float* __restrict__ data) {
  ScaleShift[((((int)blockIdx.x) * 28) + ((int)threadIdx.x))] = ((data[((((int)blockIdx.x) * 28) + ((int)threadIdx.x))] * Scale[((((((((int)blockIdx.x) % 51) * 2) + (((int)threadIdx.x) / 14)) / 51) * 3) + ((((int)blockIdx.x) + ((int)threadIdx.x)) % 3))]) + Shift[((((((((int)blockIdx.x) % 51) * 2) + (((int)threadIdx.x) / 14)) / 51) * 3) + ((((int)blockIdx.x) + ((int)threadIdx.x)) % 3))]);
}

