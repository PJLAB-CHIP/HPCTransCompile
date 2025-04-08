extern "C" __global__ void __launch_bounds__(8) default_function_kernel(float* __restrict__ compute, float* __restrict__ data) {
  compute[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))] = asinhf(data[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))]);
}

