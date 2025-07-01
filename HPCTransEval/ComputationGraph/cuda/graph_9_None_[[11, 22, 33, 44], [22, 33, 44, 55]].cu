extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_3(float* __restrict__ T_strided_slice) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 625) {
    T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = min(9.000000e+00f, T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ T_strided_slice, float* __restrict__ ph) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 625) {
    T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ph[((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) >> 3)) / 125) * 31944) + (((((((int)blockIdx.x) * 6) + (((int)threadIdx.x) >> 2)) % 250) / 25) * 1452)) + (((((((int)blockIdx.x) * 12) + (((int)threadIdx.x) >> 1)) % 50) / 5) * 44)) + (((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 10)) + 16973)];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_5(float* __restrict__ T_fast_tanh, float* __restrict__ T_strided_slice) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 625) {
    T_fast_tanh[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ((T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * (((T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * (((T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * (((T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * (((T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * (((T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * (((T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * -2.760768e-16f) + 2.000188e-13f)) + -8.604672e-11f)) + 5.122297e-08f)) + 1.485722e-05f)) + 6.372619e-04f)) + 4.893525e-03f)) / (((T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * (((T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * (((T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * 1.198258e-06f) + 1.185347e-04f)) + 2.268435e-03f)) + 4.893525e-03f));
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_4(float* __restrict__ T_strided_slice) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 625) {
    T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max(-9.000000e+00f, T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_1(float* __restrict__ T_strided_slice, float* __restrict__ ph) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 625) {
    T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ph[((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) >> 3)) / 125) * 79860) + (((((((int)blockIdx.x) * 6) + (((int)threadIdx.x) >> 2)) % 250) / 25) * 2420)) + (((((((int)blockIdx.x) * 12) + (((int)threadIdx.x) >> 1)) % 50) / 5) * 55)) + (((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 10)) + 933559)];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_2(float* __restrict__ T_strided_slice, float* __restrict__ T_strided_slice_1) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 625) {
    T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] - T_strided_slice_1[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

