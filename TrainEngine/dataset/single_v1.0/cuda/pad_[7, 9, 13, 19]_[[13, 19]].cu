extern "C" __global__ void __launch_bounds__(2) default_function_kernel(float* __restrict__ PadInput, float* __restrict__ data) {
  PadInput[((((int)blockIdx.x) * 2) + ((int)threadIdx.x))] = (((((11 <= ((int)blockIdx.x)) && (((int)blockIdx.x) < 154)) && (1 <= (((int)blockIdx.x) % 11))) && ((((((int)blockIdx.x) % 11) * 2) + ((int)threadIdx.x)) < 21)) ? data[(((((((int)blockIdx.x) / 11) * 19) + ((((int)blockIdx.x) % 11) * 2)) + ((int)threadIdx.x)) - 21)] : 0.000000e+00f);
}

