extern "C" __global__ void __launch_bounds__(16) default_function_kernel(float* __restrict__ compute, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 1)) < 147) {
    compute[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))] = fabsf(data[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))]);
  }
}

