extern "C" __global__ void __launch_bounds__(16) default_function_kernel(float* __restrict__ MirrorPadInput, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 1)) < 49) {
    MirrorPadInput[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))] = data[((((35 <= ((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 1))) ? (8 - (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 1)) / 7)) : ((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 1)) < 7) ? 0 : ((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 1)) / 7) - 1))) * 11) + (((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) == 13) ? (23 - (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14)) : (((((((int)threadIdx.x) >> 1) + ((int)blockIdx.x)) % 7) < 1) ? (1 - (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14)) : ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) - 2))))];
  }
}

