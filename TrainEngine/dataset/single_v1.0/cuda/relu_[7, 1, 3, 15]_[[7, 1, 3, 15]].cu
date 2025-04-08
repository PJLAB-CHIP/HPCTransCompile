extern "C" __global__ void __launch_bounds__(9) default_function_kernel(float* __restrict__ compute, float* __restrict__ data) {
  compute[((((int)blockIdx.x) * 9) + ((int)threadIdx.x))] = max(data[((((int)blockIdx.x) * 9) + ((int)threadIdx.x))], 0.000000e+00f);
}

