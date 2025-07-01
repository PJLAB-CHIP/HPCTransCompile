extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ A, float* __restrict__ T_transpose) {
  T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = A[(((((((int)blockIdx.x) / 20) * 20480) + ((((int)threadIdx.x) & 63) * 320)) + ((((int)blockIdx.x) % 20) * 16)) + (((int)threadIdx.x) >> 6))];
}

