extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_1(float* __restrict__ T_transpose) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 405) {
    T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = floorf(T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ T_transpose, float* __restrict__ ph) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 405) {
    T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ph[((((((int)blockIdx.x) + ((int)threadIdx.x)) % 3) * 540) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 3))];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_2(float* __restrict__ T_transpose, float* __restrict__ compute) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 405) {
    compute[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max(min(T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 1.111000e+01f), 1.100000e+00f);
  }
}

