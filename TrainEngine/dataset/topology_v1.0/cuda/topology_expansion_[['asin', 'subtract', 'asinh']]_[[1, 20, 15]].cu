extern "C" __global__ void __launch_bounds__(16) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 3)) < 9) {
    compute[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))] = asinhf((ph_0[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))] - asinf(ph_0[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))])));
  }
}

