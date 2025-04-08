extern "C" __global__ void __launch_bounds__(48) default_function_kernel(float* __restrict__ compute, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) / 12)) < 2565) {
    compute[((((int)blockIdx.x) * 48) + ((int)threadIdx.x))] = __log10f(data[((((int)blockIdx.x) * 48) + ((int)threadIdx.x))]);
  }
}

