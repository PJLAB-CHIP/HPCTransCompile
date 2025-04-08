extern "C" __global__ void __launch_bounds__(16) default_function_kernel(float* __restrict__ compute, float* __restrict__ data) {
  compute[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))] = __sinf(data[((((int)blockIdx.x) * 16) + ((int)threadIdx.x))]);
}

