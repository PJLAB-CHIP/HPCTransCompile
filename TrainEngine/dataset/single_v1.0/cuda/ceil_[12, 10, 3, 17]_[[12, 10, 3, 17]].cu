extern "C" __global__ void __launch_bounds__(20) default_function_kernel(float* __restrict__ compute, float* __restrict__ data) {
  compute[((((int)blockIdx.x) * 20) + ((int)threadIdx.x))] = ceilf(data[((((int)blockIdx.x) * 20) + ((int)threadIdx.x))]);
}

