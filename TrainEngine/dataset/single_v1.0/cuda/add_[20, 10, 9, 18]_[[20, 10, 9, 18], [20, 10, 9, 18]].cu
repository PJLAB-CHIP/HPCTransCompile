extern "C" __global__ void __launch_bounds__(50) default_function_kernel(float* __restrict__ T_add, float* __restrict__ data, float* __restrict__ data_1) {
  T_add[((((int)blockIdx.x) * 50) + ((int)threadIdx.x))] = (data[((((int)blockIdx.x) * 50) + ((int)threadIdx.x))] + data_1[((((int)blockIdx.x) * 50) + ((int)threadIdx.x))]);
}

