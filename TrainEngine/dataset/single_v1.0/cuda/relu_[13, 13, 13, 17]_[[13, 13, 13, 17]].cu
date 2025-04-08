extern "C" __global__ void __launch_bounds__(48) default_function_kernel(float* __restrict__ compute, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 48) + ((int)threadIdx.x)) < 37349) {
    compute[((((int)blockIdx.x) * 48) + ((int)threadIdx.x))] = max(data[((((int)blockIdx.x) * 48) + ((int)threadIdx.x))], 0.000000e+00f);
  }
}

