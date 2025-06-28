extern "C" __global__ void __launch_bounds__(22) default_function_kernel(float* __restrict__ compute, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 22) + ((int)threadIdx.x)) < 133) {
    compute[((((int)blockIdx.x) * 22) + ((int)threadIdx.x))] = (1.000000e+00f / (1.000000e+00f + __expf((0.000000e+00f - data[((((int)blockIdx.x) * 22) + ((int)threadIdx.x))]))));
  }
}

