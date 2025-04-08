extern "C" __global__ void __launch_bounds__(54) default_function_kernel(float* __restrict__ Scale, float* __restrict__ ScaleShift, float* __restrict__ Shift, float* __restrict__ data) {
  ScaleShift[((((int)blockIdx.x) * 54) + ((int)threadIdx.x))] = ((data[((((int)blockIdx.x) * 54) + ((int)threadIdx.x))] * Scale[((((((((int)blockIdx.x) * 9) + (((int)threadIdx.x) / 6)) % 160) / 80) * 10) + (((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 10))]) + Shift[((((((((int)blockIdx.x) * 9) + (((int)threadIdx.x) / 6)) % 160) / 80) * 10) + (((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 10))]);
}

