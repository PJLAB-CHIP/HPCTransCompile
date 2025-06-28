extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ MirrorPadInput, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) < 77) {
    MirrorPadInput[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = data[((((55 <= ((((int)blockIdx.x) * 32) + ((int)threadIdx.x))) ? (8 - (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) / 11)) : ((((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) < 11) ? 0 : ((((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) / 11) - 1))) * 8) + (((((((int)blockIdx.x) * 10) + ((int)threadIdx.x)) % 11) == 10) ? (17 - (((((int)blockIdx.x) * 10) + ((int)threadIdx.x)) % 11)) : (((((((int)blockIdx.x) * 10) + ((int)threadIdx.x)) % 11) < 2) ? (1 - (((((int)blockIdx.x) * 10) + ((int)threadIdx.x)) % 11)) : ((((((int)blockIdx.x) * 10) + ((int)threadIdx.x)) % 11) - 2))))];
  }
}

