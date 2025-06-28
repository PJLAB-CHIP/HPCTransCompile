extern "C" __global__ void __launch_bounds__(6) default_function_kernel(float* __restrict__ compute, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 3) + (((int)threadIdx.x) >> 1)) < 220) {
    compute[((((int)blockIdx.x) * 6) + ((int)threadIdx.x))] = tanhf(data[((((int)blockIdx.x) * 6) + ((int)threadIdx.x))]);
  }
}

