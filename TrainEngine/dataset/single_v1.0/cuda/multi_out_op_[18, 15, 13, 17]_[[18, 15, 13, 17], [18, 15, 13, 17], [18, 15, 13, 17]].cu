extern "C" __global__ void __launch_bounds__(48) default_function_kernel_1(float* __restrict__ compute, float* __restrict__ data, float* __restrict__ data_1) {
  if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) / 6)) < 9945) {
    compute[((((int)blockIdx.x) * 48) + ((int)threadIdx.x))] = __cosf((data[((((int)blockIdx.x) * 48) + ((int)threadIdx.x))] + data_1[((((int)blockIdx.x) * 48) + ((int)threadIdx.x))]));
  }
}

extern "C" __global__ void __launch_bounds__(64) default_function_kernel(float* __restrict__ compute, float* __restrict__ data, float* __restrict__ data_1) {
  if (((((int)blockIdx.x) * 32) + (((int)threadIdx.x) >> 1)) < 29835) {
    compute[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = sqrtf((data[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] + data_1[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))]));
  }
}

