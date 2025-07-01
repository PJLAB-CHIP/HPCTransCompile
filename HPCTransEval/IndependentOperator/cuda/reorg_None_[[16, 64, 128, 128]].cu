extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ A, float* __restrict__ T_reshape) {
  T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = A[(((((((((int)blockIdx.x) >> 10) * 4194304) + ((((int)blockIdx.x) & 255) * 4096)) + ((((int)threadIdx.x) >> 7) * 512)) + (((((int)blockIdx.x) & 1023) >> 9) * 256)) + ((((int)threadIdx.x) & 127) * 2)) + ((((int)blockIdx.x) & 511) >> 8))];
}

