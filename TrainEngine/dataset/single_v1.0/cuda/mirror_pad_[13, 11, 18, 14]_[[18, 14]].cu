extern "C" __global__ void __launch_bounds__(8) default_function_kernel(float* __restrict__ MirrorPadInput, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.x)) < 357) {
    MirrorPadInput[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))] = data[((((323 <= ((((int)blockIdx.x) * 8) + ((int)threadIdx.x))) ? (36 - (((((int)blockIdx.x) * 8) + ((int)threadIdx.x)) / 17)) : ((((((int)blockIdx.x) * 8) + ((int)threadIdx.x)) < 17) ? 0 : ((((((int)blockIdx.x) * 8) + ((int)threadIdx.x)) / 17) - 1))) * 14) + (((((((int)blockIdx.x) * 8) + ((int)threadIdx.x)) % 17) == 16) ? (29 - (((((int)blockIdx.x) * 8) + ((int)threadIdx.x)) % 17)) : (((((((int)blockIdx.x) * 8) + ((int)threadIdx.x)) % 17) < 2) ? (1 - (((((int)blockIdx.x) * 8) + ((int)threadIdx.x)) % 17)) : ((((((int)blockIdx.x) * 8) + ((int)threadIdx.x)) % 17) - 2))))];
  }
}

