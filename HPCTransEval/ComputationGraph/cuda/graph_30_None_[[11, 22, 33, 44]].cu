extern "C" __global__ void __launch_bounds__(384) default_function_kernel_3(float* __restrict__ T_strided_slice, float* __restrict__ compute) {
  compute[((int)threadIdx.x)] = __expf(T_strided_slice[((int)threadIdx.x)]);
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ T_strided_slice, float* __restrict__ ph) {
  if (((int)threadIdx.x) < 384) {
    T_strided_slice[((int)threadIdx.x)] = ph[((((((((int)threadIdx.x) / 192) * 31944) + (((((int)threadIdx.x) % 192) / 48) * 1452)) + (((((int)threadIdx.x) % 48) >> 3) * 44)) + (((int)threadIdx.x) & 7)) + 149194)];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_1(float* __restrict__ T_strided_slice) {
  if (((int)threadIdx.x) < 384) {
    T_strided_slice[((int)threadIdx.x)] = __log10f(T_strided_slice[((int)threadIdx.x)]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_2(float* __restrict__ T_strided_slice) {
  if (((int)threadIdx.x) < 384) {
    T_strided_slice[((int)threadIdx.x)] = roundf(T_strided_slice[((int)threadIdx.x)]);
  }
}

