extern "C" __global__ void __launch_bounds__(14) default_function_kernel(float* __restrict__ PadInput, float* __restrict__ data) {
  PadInput[((((int)blockIdx.x) * 14) + ((int)threadIdx.x))] = (((((1 <= ((int)blockIdx.x)) && (((int)blockIdx.x) < 4)) && (2 <= ((int)threadIdx.x))) && (((int)threadIdx.x) < 13)) ? data[(((((int)blockIdx.x) * 11) + ((int)threadIdx.x)) - 13)] : 0.000000e+00f);
}

