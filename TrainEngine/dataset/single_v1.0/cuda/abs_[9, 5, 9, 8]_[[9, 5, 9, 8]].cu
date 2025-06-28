extern "C" __global__ void __launch_bounds__(54) default_function_kernel(float* __restrict__ compute, float* __restrict__ data) {
  compute[((((int)blockIdx.x) * 54) + ((int)threadIdx.x))] = fabsf(data[((((int)blockIdx.x) * 54) + ((int)threadIdx.x))]);
}

