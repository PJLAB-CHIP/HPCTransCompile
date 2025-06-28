extern "C" __global__ void __launch_bounds__(15) default_function_kernel(float* __restrict__ MirrorPadInput, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 15) + ((int)threadIdx.x)) < 484) {
    MirrorPadInput[((((int)blockIdx.x) * 15) + ((int)threadIdx.x))] = data[((((88 <= ((((int)blockIdx.x) * 3) + (((int)threadIdx.x) / 5))) ? (38 - (((((int)blockIdx.x) * 15) + ((int)threadIdx.x)) / 22)) : ((((((int)blockIdx.x) * 15) + ((int)threadIdx.x)) < 22) ? 0 : ((((((int)blockIdx.x) * 15) + ((int)threadIdx.x)) / 22) - 1))) * 19) + (((((((int)blockIdx.x) * 15) + ((int)threadIdx.x)) % 22) == 21) ? (39 - (((((int)blockIdx.x) * 15) + ((int)threadIdx.x)) % 22)) : (((((((int)blockIdx.x) * 15) + ((int)threadIdx.x)) % 22) < 2) ? (1 - (((((int)blockIdx.x) * 15) + ((int)threadIdx.x)) % 22)) : ((((((int)blockIdx.x) * 15) + ((int)threadIdx.x)) % 22) - 2))))];
  }
}

