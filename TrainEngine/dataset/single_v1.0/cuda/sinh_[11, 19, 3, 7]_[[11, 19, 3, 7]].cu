extern "C" __global__ void __launch_bounds__(11) default_function_kernel(float* __restrict__ compute, float* __restrict__ data) {
  compute[((((int)blockIdx.x) * 11) + ((int)threadIdx.x))] = sinhf(data[((((int)blockIdx.x) * 11) + ((int)threadIdx.x))]);
}

