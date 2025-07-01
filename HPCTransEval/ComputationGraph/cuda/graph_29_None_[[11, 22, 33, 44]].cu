extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_2(float* __restrict__ T_strided_slice) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 625) {
    T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 1.000000e+01f), 0.000000e+00f);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_3(float* __restrict__ T_strided_slice) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 625) {
    T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = tanhf(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_4(float* __restrict__ T_strided_slice, float* __restrict__ T_transpose) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 625) {
    T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_strided_slice[(((((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 10) * 1000) + ((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) >> 3)) / 125) * 100)) + (((((((int)blockIdx.x) * 12) + (((int)threadIdx.x) >> 1)) % 50) / 5) * 10)) + ((((((int)blockIdx.x) * 6) + (((int)threadIdx.x) >> 2)) % 250) / 25))];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ T_strided_slice, float* __restrict__ ph) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 625) {
    T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ph[((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) >> 3)) / 125) * 31944) + (((((((int)blockIdx.x) * 6) + (((int)threadIdx.x) >> 2)) % 250) / 25) * 1452)) + (((((((int)blockIdx.x) * 12) + (((int)threadIdx.x) >> 1)) % 50) / 5) * 44)) + (((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 10)) + 16973)];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_1(float* __restrict__ T_strided_slice) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 625) {
    T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = __log2f(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

