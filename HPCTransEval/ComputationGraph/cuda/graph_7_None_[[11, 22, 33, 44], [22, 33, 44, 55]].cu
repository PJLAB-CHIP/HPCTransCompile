extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_1(float* __restrict__ T_strided_slice, float* __restrict__ ph) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 625) {
    T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ph[((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) >> 3)) / 125) * 79860) + (((((((int)blockIdx.x) * 6) + (((int)threadIdx.x) >> 2)) % 250) / 25) * 2420)) + (((((((int)blockIdx.x) * 12) + (((int)threadIdx.x) >> 1)) % 50) / 5) * 55)) + (((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 10)) + 933559)];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_3(float* __restrict__ T_fast_exp, float* __restrict__ T_strided_slice) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 625) {
      int v_ = ((int)(floorf(((max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) + 1.270000e+02f)) << 23;
    T_fast_exp[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max(((*(float *)(&(v_))) * ((((((((((((((1.987569e-04f * (max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.398200e-03f) * (max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 8.333452e-03f) * (max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 4.166580e-02f) * (max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.666667e-01f) * (max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 5.000000e-01f) * (max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) * (max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + (max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.000000e+00f)), T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ T_strided_slice, float* __restrict__ ph) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 625) {
    T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ph[((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) >> 3)) / 125) * 31944) + (((((((int)blockIdx.x) * 6) + (((int)threadIdx.x) >> 2)) % 250) / 25) * 1452)) + (((((((int)blockIdx.x) * 12) + (((int)threadIdx.x) >> 1)) % 50) / 5) * 44)) + (((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 10)) + 16973)];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_2(float* __restrict__ T_strided_slice, float* __restrict__ T_strided_slice_1) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 625) {
    T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_strided_slice_1[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

