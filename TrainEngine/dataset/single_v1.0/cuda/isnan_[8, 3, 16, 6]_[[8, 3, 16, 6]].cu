extern "C" __global__ void __launch_bounds__(12) default_function_kernel(signed char* __restrict__ compute, float* __restrict__ data) {
  compute[((((int)blockIdx.x) * 12) + ((int)threadIdx.x))] = ((signed char)(data[((((int)blockIdx.x) * 12) + ((int)threadIdx.x))] != data[((((int)blockIdx.x) * 12) + ((int)threadIdx.x))]));
}

