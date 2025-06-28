extern "C" __global__ void __launch_bounds__(63) default_function_kernel(signed char* __restrict__ compute, float* __restrict__ data) {
  compute[((((int)blockIdx.x) * 63) + ((int)threadIdx.x))] = ((signed char)(data[((((int)blockIdx.x) * 63) + ((int)threadIdx.x))] != data[((((int)blockIdx.x) * 63) + ((int)threadIdx.x))]));
}

