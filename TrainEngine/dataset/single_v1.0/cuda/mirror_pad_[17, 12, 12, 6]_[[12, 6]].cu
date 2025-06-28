extern "C" __global__ void __launch_bounds__(64) default_function_kernel(float* __restrict__ MirrorPadInput, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) < 135) {
    MirrorPadInput[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = data[((((117 <= ((((int)blockIdx.x) * 64) + ((int)threadIdx.x))) ? (24 - (((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) / 9)) : ((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) < 9) ? 0 : ((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) / 9) - 1))) * 6) + ((((((int)blockIdx.x) + ((int)threadIdx.x)) % 9) == 8) ? (13 - ((((int)blockIdx.x) + ((int)threadIdx.x)) % 9)) : ((((((int)blockIdx.x) + ((int)threadIdx.x)) % 9) < 2) ? (1 - ((((int)blockIdx.x) + ((int)threadIdx.x)) % 9)) : (((((int)blockIdx.x) + ((int)threadIdx.x)) % 9) - 2))))];
  }
}

