extern "C" __global__ void __launch_bounds__(28) default_function_kernel(float* __restrict__ PadInput, float* __restrict__ data) {
  PadInput[((int)threadIdx.x)] = (((((7 <= ((int)threadIdx.x)) && (((int)threadIdx.x) < 14)) && (2 <= (((int)threadIdx.x) % 7))) && ((((int)threadIdx.x) % 7) < 6)) ? data[((((((int)threadIdx.x) / 7) * 4) + (((int)threadIdx.x) % 7)) - 6)] : 0.000000e+00f);
}

