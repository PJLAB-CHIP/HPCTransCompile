extern "C" __global__ void __launch_bounds__(12) default_function_kernel(float* __restrict__ T_add, float* __restrict__ data, float* __restrict__ data_1) {
  T_add[((int)threadIdx.x)] = (data[((int)threadIdx.x)] + data_1[((int)threadIdx.x)]);
}

