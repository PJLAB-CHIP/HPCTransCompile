extern "C" __global__ void __launch_bounds__(56) default_function_kernel(float* __restrict__ Scale, float* __restrict__ ScaleShift, float* __restrict__ Shift, float* __restrict__ data) {
  ScaleShift[((((int)blockIdx.x) * 56) + ((int)threadIdx.x))] = ((data[((((int)blockIdx.x) * 56) + ((int)threadIdx.x))] * Scale[((((((((int)blockIdx.x) % 55) * 2) + (((int)threadIdx.x) / 28)) / 55) * 11) + ((((int)blockIdx.x) + ((int)threadIdx.x)) % 11))]) + Shift[((((((((int)blockIdx.x) % 55) * 2) + (((int)threadIdx.x) / 28)) / 55) * 11) + ((((int)blockIdx.x) + ((int)threadIdx.x)) % 11))]);
}

