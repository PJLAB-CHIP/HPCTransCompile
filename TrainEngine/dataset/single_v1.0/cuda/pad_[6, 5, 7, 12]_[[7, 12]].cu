extern "C" __global__ void __launch_bounds__(16) default_function_kernel(float* __restrict__ PadInput, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 1)) < 75) {
    PadInput[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))] = (((((15 <= ((((int)blockIdx.x) * 16) + ((int)threadIdx.x))) && (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 3)) < 15)) && (2 <= ((((int)blockIdx.x) + ((int)threadIdx.x)) % 15))) && (((((int)blockIdx.x) + ((int)threadIdx.x)) % 15) < 14)) ? data[((((((((int)blockIdx.x) * 16) + ((int)threadIdx.x)) / 15) * 12) + ((((int)blockIdx.x) + ((int)threadIdx.x)) % 15)) - 14)] : 0.000000e+00f);
  }
}

