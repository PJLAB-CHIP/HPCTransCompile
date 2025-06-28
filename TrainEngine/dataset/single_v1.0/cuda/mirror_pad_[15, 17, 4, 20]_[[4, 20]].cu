extern "C" __global__ void __launch_bounds__(5) default_function_kernel(float* __restrict__ MirrorPadInput, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 5) + ((int)threadIdx.x)) < 161) {
    MirrorPadInput[((((int)blockIdx.x) * 5) + ((int)threadIdx.x))] = data[((((23 <= ((int)blockIdx.x)) ? (8 - (((((int)blockIdx.x) * 5) + ((int)threadIdx.x)) / 23)) : ((((((int)blockIdx.x) * 5) + ((int)threadIdx.x)) < 23) ? 0 : ((((((int)blockIdx.x) * 5) + ((int)threadIdx.x)) / 23) - 1))) * 20) + (((((((int)blockIdx.x) * 5) + ((int)threadIdx.x)) % 23) == 22) ? (41 - (((((int)blockIdx.x) * 5) + ((int)threadIdx.x)) % 23)) : (((((((int)blockIdx.x) * 5) + ((int)threadIdx.x)) % 23) < 2) ? (1 - (((((int)blockIdx.x) * 5) + ((int)threadIdx.x)) % 23)) : ((((((int)blockIdx.x) * 5) + ((int)threadIdx.x)) % 23) - 2))))];
  }
}

