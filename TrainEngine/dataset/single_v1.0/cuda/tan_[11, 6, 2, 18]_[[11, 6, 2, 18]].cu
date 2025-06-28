extern "C" __global__ void __launch_bounds__(22) default_function_kernel(float* __restrict__ compute, float* __restrict__ data) {
  compute[((((int)blockIdx.x) * 22) + ((int)threadIdx.x))] = tanf(data[((((int)blockIdx.x) * 22) + ((int)threadIdx.x))]);
}

