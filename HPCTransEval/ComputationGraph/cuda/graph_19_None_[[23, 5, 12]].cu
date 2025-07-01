extern "C" __global__ void __launch_bounds__(765) default_function_kernel(float* __restrict__ T_strided_slice, float* __restrict__ ph) {
  if (((int)threadIdx.x) < 168) {
    T_strided_slice[((int)threadIdx.x)] = ph[(((((((int)threadIdx.x) / 21) * 60) + (((((int)threadIdx.x) % 21) / 7) * 12)) + (((int)threadIdx.x) % 7)) + 61)];
  }
}

extern "C" __global__ void __launch_bounds__(168) default_function_kernel_2(float* __restrict__ T_strided_slice, float* __restrict__ compute) {
  compute[((int)threadIdx.x)] = __expf(T_strided_slice[((int)threadIdx.x)]);
}

extern "C" __global__ void __launch_bounds__(765) default_function_kernel_1(float* __restrict__ T_strided_slice) {
  if (((int)threadIdx.x) < 168) {
      int v_ = ((int)(floorf(((max(min(T_strided_slice[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) + 1.270000e+02f)) << 23;
    T_strided_slice[((int)threadIdx.x)] = max(((*(float *)(&(v_))) * ((((((((((((((1.987569e-04f * (max(min(T_strided_slice[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_strided_slice[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.398200e-03f) * (max(min(T_strided_slice[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_strided_slice[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 8.333452e-03f) * (max(min(T_strided_slice[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_strided_slice[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 4.166580e-02f) * (max(min(T_strided_slice[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_strided_slice[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.666667e-01f) * (max(min(T_strided_slice[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_strided_slice[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 5.000000e-01f) * (max(min(T_strided_slice[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_strided_slice[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) * (max(min(T_strided_slice[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_strided_slice[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + (max(min(T_strided_slice[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_strided_slice[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.000000e+00f)), T_strided_slice[((int)threadIdx.x)]);
  }
}

