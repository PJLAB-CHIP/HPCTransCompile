extern "C" __global__ void __launch_bounds__(8) default_function_kernel(float* __restrict__ PadInput, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.x)) < 35) {
    PadInput[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))] = (((((7 <= ((((int)blockIdx.x) * 8) + ((int)threadIdx.x))) && (((((int)blockIdx.x) * 8) + ((int)threadIdx.x)) < 21)) && (2 <= ((((int)blockIdx.x) + ((int)threadIdx.x)) % 7))) && (((((int)blockIdx.x) + ((int)threadIdx.x)) % 7) < 6)) ? data[((((((((int)blockIdx.x) * 8) + ((int)threadIdx.x)) / 7) * 4) + ((((int)blockIdx.x) + ((int)threadIdx.x)) % 7)) - 6)] : 0.000000e+00f);
  }
}

