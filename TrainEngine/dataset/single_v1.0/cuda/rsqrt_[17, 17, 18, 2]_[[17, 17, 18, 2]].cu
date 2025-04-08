extern "C" __global__ void __launch_bounds__(50) default_function_kernel(float* __restrict__ compute, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 25) + (((int)threadIdx.x) >> 1)) < 5202) {
    compute[((((int)blockIdx.x) * 50) + ((int)threadIdx.x))] = (1.000000e+00f / sqrtf(data[((((int)blockIdx.x) * 50) + ((int)threadIdx.x))]));
  }
}

