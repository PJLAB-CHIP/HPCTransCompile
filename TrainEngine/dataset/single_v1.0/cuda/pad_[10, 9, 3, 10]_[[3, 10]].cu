extern "C" __global__ void __launch_bounds__(2) default_function_kernel(float* __restrict__ PadInput, float* __restrict__ data) {
  PadInput[((((int)blockIdx.x) * 2) + ((int)threadIdx.x))] = (((((13 <= ((((int)blockIdx.x) * 2) + ((int)threadIdx.x))) && (((int)blockIdx.x) < 26)) && (2 <= (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 13))) && ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 13) < 12)) ? data[((((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) / 13) * 10) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 13)) - 12)] : 0.000000e+00f);
}

