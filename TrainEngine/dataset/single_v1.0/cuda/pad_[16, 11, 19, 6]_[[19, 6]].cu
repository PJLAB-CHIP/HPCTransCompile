extern "C" __global__ void __launch_bounds__(22) default_function_kernel(float* __restrict__ PadInput, float* __restrict__ data) {
  PadInput[((((int)blockIdx.x) * 22) + ((int)threadIdx.x))] = (((((9 <= ((((int)blockIdx.x) * 22) + ((int)threadIdx.x))) && (((((int)blockIdx.x) * 11) + (((int)threadIdx.x) >> 1)) < 90)) && (2 <= (((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 9))) && ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 9) < 8)) ? data[((((((((int)blockIdx.x) * 22) + ((int)threadIdx.x)) / 9) * 6) + (((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 9)) - 8)] : 0.000000e+00f);
}

