extern "C" __global__ void __launch_bounds__(50) default_function_kernel(float* __restrict__ compute, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 10) + (((int)threadIdx.x) / 5)) < 1521) {
    compute[((((int)blockIdx.x) * 50) + ((int)threadIdx.x))] = ceilf(data[((((int)blockIdx.x) * 50) + ((int)threadIdx.x))]);
  }
}

