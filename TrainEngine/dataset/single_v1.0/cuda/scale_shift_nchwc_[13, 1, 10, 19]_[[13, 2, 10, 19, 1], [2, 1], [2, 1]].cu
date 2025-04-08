extern "C" __global__ void __launch_bounds__(20) default_function_kernel(float* __restrict__ Scale, float* __restrict__ ScaleShift, float* __restrict__ Shift, float* __restrict__ data) {
  ScaleShift[((((int)blockIdx.x) * 20) + ((int)threadIdx.x))] = ((data[((((int)blockIdx.x) * 20) + ((int)threadIdx.x))] * Scale[((((((int)blockIdx.x) % 19) * 2) + (((int)threadIdx.x) / 10)) / 19)]) + Shift[((((((int)blockIdx.x) % 19) * 2) + (((int)threadIdx.x) / 10)) / 19)]);
}

