extern "C" __global__ void __launch_bounds__(28) default_function_kernel(float* __restrict__ PadInput, float* __restrict__ data) {
  PadInput[((((int)blockIdx.x) * 28) + ((int)threadIdx.x))] = (((((5 <= ((((int)blockIdx.x) * 14) + (((int)threadIdx.x) >> 1))) && (((((int)blockIdx.x) * 7) + (((int)threadIdx.x) >> 2)) < 30)) && (1 <= (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 1)) % 5))) && ((((((int)blockIdx.x) * 8) + ((int)threadIdx.x)) % 10) < 9)) ? data[((((((((int)blockIdx.x) * 14) + (((int)threadIdx.x) >> 1)) / 5) * 7) + (((((int)blockIdx.x) * 8) + ((int)threadIdx.x)) % 10)) - 9)] : 0.000000e+00f);
}

