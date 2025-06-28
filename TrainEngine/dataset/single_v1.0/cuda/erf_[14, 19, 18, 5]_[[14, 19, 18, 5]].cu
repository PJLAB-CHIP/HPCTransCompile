extern "C" __global__ void __launch_bounds__(35) default_function_kernel(float* __restrict__ compute, float* __restrict__ data) {
  compute[((((int)blockIdx.x) * 35) + ((int)threadIdx.x))] = erff(data[((((int)blockIdx.x) * 35) + ((int)threadIdx.x))]);
}

