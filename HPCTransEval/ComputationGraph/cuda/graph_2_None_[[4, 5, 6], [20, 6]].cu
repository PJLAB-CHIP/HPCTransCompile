extern "C" __global__ void __launch_bounds__(120) default_function_kernel_7(float* __restrict__ T_reshape, float* __restrict__ T_softmax_maxelem, float* __restrict__ T_softmax_norm) {
  T_softmax_norm[((int)threadIdx.x)] = (T_reshape[((int)threadIdx.x)] / T_softmax_maxelem[(((int)threadIdx.x) / 6)]);
}

extern "C" __global__ void __launch_bounds__(20) default_function_kernel_3(float* __restrict__ T_reshape, float* __restrict__ T_softmax_maxelem) {
  T_softmax_maxelem[((int)threadIdx.x)] = -3.402823e+38f;
  for (int k = 0; k < 6; ++k) {
    T_softmax_maxelem[((int)threadIdx.x)] = max(T_softmax_maxelem[((int)threadIdx.x)], T_reshape[((((int)threadIdx.x) * 6) + k)]);
  }
}

extern "C" __global__ void __launch_bounds__(120) default_function_kernel_2(float* __restrict__ T_reshape) {
  T_reshape[((int)threadIdx.x)] = max(T_reshape[((int)threadIdx.x)], 0.000000e+00f);
}

extern "C" __global__ void __launch_bounds__(120) default_function_kernel(float* __restrict__ T_reshape, float* __restrict__ ph) {
  T_reshape[((int)threadIdx.x)] = ph[((int)threadIdx.x)];
}

extern "C" __global__ void __launch_bounds__(20) default_function_kernel_6(float* __restrict__ T_reshape, float* __restrict__ T_softmax_maxelem) {
  T_softmax_maxelem[((int)threadIdx.x)] = 0.000000e+00f;
  for (int k = 0; k < 6; ++k) {
    T_softmax_maxelem[((int)threadIdx.x)] = (T_softmax_maxelem[((int)threadIdx.x)] + T_reshape[((((int)threadIdx.x) * 6) + k)]);
  }
}

extern "C" __global__ void __launch_bounds__(120) default_function_kernel_1(float* __restrict__ T_reshape, float* __restrict__ ph) {
  T_reshape[((int)threadIdx.x)] = (T_reshape[((int)threadIdx.x)] - ph[((int)threadIdx.x)]);
}

extern "C" __global__ void __launch_bounds__(120) default_function_kernel_4(float* __restrict__ T_reshape, float* __restrict__ T_softmax_maxelem) {
  T_reshape[((int)threadIdx.x)] = (T_reshape[((int)threadIdx.x)] - T_softmax_maxelem[(((int)threadIdx.x) / 6)]);
}

extern "C" __global__ void __launch_bounds__(120) default_function_kernel_5(float* __restrict__ T_reshape) {
    int v_ = ((int)(floorf(((max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) + 1.270000e+02f)) << 23;
  T_reshape[((int)threadIdx.x)] = max(((*(float *)(&(v_))) * ((((((((((((((1.987569e-04f * (max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.398200e-03f) * (max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 8.333452e-03f) * (max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 4.166580e-02f) * (max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.666667e-01f) * (max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 5.000000e-01f) * (max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) * (max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + (max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((int)threadIdx.x)], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.000000e+00f)), T_reshape[((int)threadIdx.x)]);
}

