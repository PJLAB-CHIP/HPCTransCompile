extern "C" __global__ void __launch_bounds__(52) default_function_kernel(float* __restrict__ compute, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 26) + (((int)threadIdx.x) >> 1)) < 1197) {
    compute[((((int)blockIdx.x) * 52) + ((int)threadIdx.x))] = (data[((((int)blockIdx.x) * 52) + ((int)threadIdx.x))] * -1.000000e+00f);
  }
}

