extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ A, float* __restrict__ T_reshape) {
  T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = A[((((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) / 5) * 320) + (((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) >> 1)) % 160) / 5) * 10)) + (((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 10))];
}

