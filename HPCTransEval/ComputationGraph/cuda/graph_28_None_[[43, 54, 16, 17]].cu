extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ T_transpose, float* __restrict__ ph) {
  if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 7)) < 15) {
    T_transpose[(((((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) / 3) * 192) + (((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 3)) % 24) / 3) * 24)) + ((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 2)) % 6) * 4)) + (((int)threadIdx.x) & 3))] = ph[((((((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 2)) % 6) * 14688) + (((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 3)) % 24) / 3) * 272)) + ((((int)threadIdx.x) & 3) * 17)) + (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) / 3)) + 29699)];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_2(float* __restrict__ T_transpose) {
  if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 7)) < 15) {
    T_transpose[(((((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) / 3) * 192) + (((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 3)) % 24) / 3) * 24)) + ((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 2)) % 6) * 4)) + (((int)threadIdx.x) & 3))] = __log2f(T_transpose[(((((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) / 3) * 192) + (((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 3)) % 24) / 3) * 24)) + ((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 2)) % 6) * 4)) + (((int)threadIdx.x) & 3))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_4(float* __restrict__ T_fast_exp, float* __restrict__ T_strided_slice) {
  if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 7)) < 15) {
      int v_ = ((int)(floorf(((max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) + 1.270000e+02f)) << 23;
    T_fast_exp[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max(((*(float *)(&(v_))) * ((((((((((((((1.987569e-04f * (max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.398200e-03f) * (max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 8.333452e-03f) * (max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 4.166580e-02f) * (max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.666667e-01f) * (max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 5.000000e-01f) * (max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) * (max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + (max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.000000e+00f)), T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_1(float* __restrict__ T_transpose) {
  if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 7)) < 15) {
    T_transpose[(((((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) / 3) * 192) + (((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 3)) % 24) / 3) * 24)) + ((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 2)) % 6) * 4)) + (((int)threadIdx.x) & 3))] = __cosf(T_transpose[(((((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) / 3) * 192) + (((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 3)) % 24) / 3) * 24)) + ((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 2)) % 6) * 4)) + (((int)threadIdx.x) & 3))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_3(float* __restrict__ T_strided_slice, float* __restrict__ T_transpose) {
  if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 7)) < 15) {
    T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_transpose[(((((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) / 3) * 192) + (((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 3)) % 24) / 3) * 24)) + ((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 2)) % 6) * 4)) + (((int)threadIdx.x) & 3))];
  }
}

