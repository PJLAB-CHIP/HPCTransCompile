extern "C" __global__ void __launch_bounds__(63) default_function_kernel(float* __restrict__ DilatedInput, float* __restrict__ data) {
  DilatedInput[((((int)blockIdx.x) * 63) + ((int)threadIdx.x))] = data[((((int)blockIdx.x) * 63) + ((int)threadIdx.x))];
}

