extern "C" __global__ void __launch_bounds__(7) default_function_kernel(float* __restrict__ MirrorPadInput, float* __restrict__ data) {
  MirrorPadInput[((((int)blockIdx.x) * 7) + ((int)threadIdx.x))] = data[((((96 <= ((((int)blockIdx.x) * 7) + ((int)threadIdx.x))) ? (22 - (((((int)blockIdx.x) * 7) + ((int)threadIdx.x)) >> 3)) : ((((((int)blockIdx.x) * 7) + ((int)threadIdx.x)) < 8) ? 0 : ((((((int)blockIdx.x) * 7) + ((int)threadIdx.x)) >> 3) - 1))) * 5) + (((((((int)blockIdx.x) * 7) + ((int)threadIdx.x)) & 7) == 7) ? (11 - (((((int)blockIdx.x) * 7) + ((int)threadIdx.x)) & 7)) : (((((((int)blockIdx.x) * 7) + ((int)threadIdx.x)) & 7) < 2) ? (1 - (((((int)blockIdx.x) * 7) + ((int)threadIdx.x)) & 7)) : ((((((int)blockIdx.x) * 7) + ((int)threadIdx.x)) & 7) - 2))))];
}

