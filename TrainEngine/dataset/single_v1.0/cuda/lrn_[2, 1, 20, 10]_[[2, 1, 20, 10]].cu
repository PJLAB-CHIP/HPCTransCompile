extern "C" __global__ void __launch_bounds__(32) default_function_kernel_1(float* __restrict__ T_divide, float* __restrict__ data, float* __restrict__ tensor) {
  if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 4)) < 25) {
    T_divide[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (data[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] / powf((2.000000e+00f + (1.000000e-04f * tensor[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))])), 7.500000e-01f));
  }
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ data, float* __restrict__ tensor) {
  if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 4)) < 25) {
    tensor[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = 0.000000e+00f;
  }
  if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 4)) < 25) {
    tensor[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (tensor[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + (data[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] * data[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]));
  }
}

