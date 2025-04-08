extern "C" __global__ void __launch_bounds__(7) default_function_kernel(float* __restrict__ MirrorPadInput, float* __restrict__ data) {
  MirrorPadInput[((((int)blockIdx.x) * 7) + ((int)threadIdx.x))] = data[((((80 <= ((((int)blockIdx.x) * 7) + ((int)threadIdx.x))) ? (8 - (((((int)blockIdx.x) * 7) + ((int)threadIdx.x)) >> 4)) : ((((((int)blockIdx.x) * 7) + ((int)threadIdx.x)) < 16) ? 0 : ((((((int)blockIdx.x) * 7) + ((int)threadIdx.x)) >> 4) - 1))) * 13) + (((((((int)blockIdx.x) * 7) + ((int)threadIdx.x)) & 15) == 15) ? (27 - (((((int)blockIdx.x) * 7) + ((int)threadIdx.x)) & 15)) : (((((((int)blockIdx.x) * 7) + ((int)threadIdx.x)) & 15) < 2) ? (1 - (((((int)blockIdx.x) * 7) + ((int)threadIdx.x)) & 15)) : ((((((int)blockIdx.x) * 7) + ((int)threadIdx.x)) & 15) - 2))))];
}

