extern "C" __global__ void __launch_bounds__(840) default_function_kernel_3(float* __restrict__ T_reshape, float* __restrict__ T_transpose) {
  T_transpose[((int)threadIdx.x)] = T_reshape[(((((((int)threadIdx.x) % 28) / 7) * 210) + ((((int)threadIdx.x) / 28) * 7)) + (((int)threadIdx.x) % 7))];
}

extern "C" __global__ void __launch_bounds__(840) default_function_kernel_1(float* __restrict__ T_reshape) {
  T_reshape[((int)threadIdx.x)] = max(min(T_reshape[((int)threadIdx.x)], 9.999000e+01f), 1.111000e+01f);
}

extern "C" __global__ void __launch_bounds__(840) default_function_kernel_2(float* __restrict__ T_reshape) {
    int v_ = ((int)(floorf(((max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) + 1.270000e+02f)) << 23;
  T_reshape[((int)threadIdx.x)] = max(((*(float *)(&(v_))) * ((((((((((((((1.987569e-04f * (max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.398200e-03f) * (max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 8.333452e-03f) * (max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 4.166580e-02f) * (max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.666667e-01f) * (max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 5.000000e-01f) * (max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) * (max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + (max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.000000e+00f)), T_reshape[((int)threadIdx.x)]);
}

extern "C" __global__ void __launch_bounds__(840) default_function_kernel(float* __restrict__ T_reshape, float* __restrict__ ph) {
  T_reshape[((int)threadIdx.x)] = ph[((int)threadIdx.x)];
}

