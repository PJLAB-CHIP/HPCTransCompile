extern "C" __global__ void __launch_bounds__(54) default_function_kernel(float* __restrict__ PadInput, float* __restrict__ data) {
  PadInput[((((int)blockIdx.x) * 54) + ((int)threadIdx.x))] = (((((1 <= ((((int)blockIdx.x) * 6) + (((int)threadIdx.x) / 9))) && (((((int)blockIdx.x) * 3) + (((int)threadIdx.x) / 18)) < 5)) && (2 <= (((int)threadIdx.x) % 9))) && ((((int)threadIdx.x) % 9) < 8)) ? data[((((((int)blockIdx.x) * 36) + ((((int)threadIdx.x) / 9) * 6)) + (((int)threadIdx.x) % 9)) - 8)] : 0.000000e+00f);
}

