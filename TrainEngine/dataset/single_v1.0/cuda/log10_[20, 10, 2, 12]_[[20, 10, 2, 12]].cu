extern "C" __global__ void __launch_bounds__(24) default_function_kernel(float* __restrict__ compute, float* __restrict__ data) {
  compute[((((int)blockIdx.x) * 24) + ((int)threadIdx.x))] = __log10f(data[((((int)blockIdx.x) * 24) + ((int)threadIdx.x))]);
}

