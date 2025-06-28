extern "C" __global__ void __launch_bounds__(30) default_function_kernel(float* __restrict__ compute, float* __restrict__ data) {
  compute[((int)threadIdx.x)] = fabsf(data[((int)threadIdx.x)]);
}

