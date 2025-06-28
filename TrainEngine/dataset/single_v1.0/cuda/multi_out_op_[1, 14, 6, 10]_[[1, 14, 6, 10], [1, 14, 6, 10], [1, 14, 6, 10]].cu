extern "C" __global__ void __launch_bounds__(64) default_function_kernel_1(float* __restrict__ compute, float* __restrict__ data, float* __restrict__ data_1) {
  if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 3)) < 105) {
    compute[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = __cosf((data[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] + data_1[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))]));
  }
}

extern "C" __global__ void __launch_bounds__(21) default_function_kernel(float* __restrict__ compute, float* __restrict__ data, float* __restrict__ data_1) {
  compute[((((int)blockIdx.x) * 21) + ((int)threadIdx.x))] = sqrtf((data[((((int)blockIdx.x) * 21) + ((int)threadIdx.x))] + data_1[((((int)blockIdx.x) * 21) + ((int)threadIdx.x))]));
}

