extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_5(float* __restrict__ T_reshape, float* __restrict__ T_transpose) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 1035) {
    T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_reshape[(((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 1380) * 3) + (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 345))];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ T_reshape, float* __restrict__ ph) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 1035) {
    T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ph[(((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 3) * 12) + (((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 12))];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_3(float* __restrict__ T_reshape) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 1035) {
    T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ((T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * (((T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * (((T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * (((T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * (((T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * (((T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * (((T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * -2.760768e-16f) + 2.000188e-13f)) + -8.604672e-11f)) + 5.122297e-08f)) + 1.485722e-05f)) + 6.372619e-04f)) + 4.893525e-03f)) / (((T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * (((T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * (((T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * 1.198258e-06f) + 1.185347e-04f)) + 2.268435e-03f)) + 4.893525e-03f));
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_1(float* __restrict__ T_reshape) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 1035) {
    T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = min(9.000000e+00f, T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_2(float* __restrict__ T_reshape) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 1035) {
    T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max(-9.000000e+00f, T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_4(float* __restrict__ T_reshape, float* __restrict__ T_reshape_1) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 1035) {
    T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_reshape_1[((((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 15) * 60) + (((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 60) / 3) * 3)) + ((((int)blockIdx.x) + ((int)threadIdx.x)) % 3))];
  }
}

