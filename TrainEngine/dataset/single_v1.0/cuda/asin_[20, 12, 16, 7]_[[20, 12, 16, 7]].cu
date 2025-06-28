extern "C" __global__ void __launch_bounds__(21) default_function_kernel(float* __restrict__ compute, float* __restrict__ data) {
  compute[((((int)blockIdx.x) * 21) + ((int)threadIdx.x))] = asinf(data[((((int)blockIdx.x) * 21) + ((int)threadIdx.x))]);
}

