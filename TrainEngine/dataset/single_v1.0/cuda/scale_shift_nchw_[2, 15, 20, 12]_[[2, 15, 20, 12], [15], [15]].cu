extern "C" __global__ void __launch_bounds__(24) default_function_kernel(float* __restrict__ Scale, float* __restrict__ ScaleShift, float* __restrict__ Shift, float* __restrict__ data) {
  ScaleShift[((((int)blockIdx.x) * 24) + ((int)threadIdx.x))] = ((data[((((int)blockIdx.x) * 24) + ((int)threadIdx.x))] * Scale[((((int)blockIdx.x) % 150) / 10)]) + Shift[((((int)blockIdx.x) % 150) / 10)]);
}

