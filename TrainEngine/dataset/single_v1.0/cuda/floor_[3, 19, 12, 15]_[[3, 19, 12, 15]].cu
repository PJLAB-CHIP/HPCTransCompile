extern "C" __global__ void __launch_bounds__(27) default_function_kernel(float* __restrict__ compute, float* __restrict__ data) {
  compute[((((int)blockIdx.x) * 27) + ((int)threadIdx.x))] = floorf(data[((((int)blockIdx.x) * 27) + ((int)threadIdx.x))]);
}

