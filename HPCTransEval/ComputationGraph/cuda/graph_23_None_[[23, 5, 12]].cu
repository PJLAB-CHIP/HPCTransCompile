extern "C" __global__ void __launch_bounds__(765) default_function_kernel_2(float* __restrict__ T_reverse_sequence) {
    int v_ = ((int)(floorf(((max(min(T_reverse_sequence[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) + 1.270000e+02f)) << 23;
  T_reverse_sequence[((int)threadIdx.x)] = max(((*(float *)(&(v_))) * ((((((((((((((1.987569e-04f * (max(min(T_reverse_sequence[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reverse_sequence[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.398200e-03f) * (max(min(T_reverse_sequence[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reverse_sequence[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 8.333452e-03f) * (max(min(T_reverse_sequence[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reverse_sequence[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 4.166580e-02f) * (max(min(T_reverse_sequence[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reverse_sequence[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.666667e-01f) * (max(min(T_reverse_sequence[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reverse_sequence[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 5.000000e-01f) * (max(min(T_reverse_sequence[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reverse_sequence[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) * (max(min(T_reverse_sequence[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reverse_sequence[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + (max(min(T_reverse_sequence[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reverse_sequence[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.000000e+00f)), T_reverse_sequence[((int)threadIdx.x)]);
}

extern "C" __global__ void __launch_bounds__(765) default_function_kernel_3(float* __restrict__ T_reverse_sequence, float* __restrict__ T_transpose) {
  T_transpose[((int)threadIdx.x)] = T_reverse_sequence[(((((int)threadIdx.x) % 17) * 45) + (((int)threadIdx.x) / 17))];
}

extern "C" __global__ void __launch_bounds__(765) default_function_kernel_1(float* __restrict__ T_reverse_sequence, float* __restrict__ T_strided_slice) {
  T_reverse_sequence[((int)threadIdx.x)] = T_strided_slice[(((((((int)threadIdx.x) / 45) * 45) + (((int)threadIdx.x) % 9)) + 36) - (((((int)threadIdx.x) % 45) / 9) * 9))];
}

extern "C" __global__ void __launch_bounds__(765) default_function_kernel(float* __restrict__ T_strided_slice, float* __restrict__ ph) {
  T_strided_slice[((int)threadIdx.x)] = ph[((((((int)threadIdx.x) / 9) * 12) + (((int)threadIdx.x) % 9)) + 61)];
}

