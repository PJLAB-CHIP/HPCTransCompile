extern "C" __global__ void __launch_bounds__(10) default_function_kernel(float* __restrict__ compute, float* __restrict__ data) {
  compute[((((int)blockIdx.x) * 10) + ((int)threadIdx.x))] = __cosf(data[((((int)blockIdx.x) * 10) + ((int)threadIdx.x))]);
}

