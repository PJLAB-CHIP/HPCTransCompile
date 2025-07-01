extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ T_transpose, float* __restrict__ ph) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 3105) {
    T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ph[((((((((int)blockIdx.x) * 12) + ((int)threadIdx.x)) % 23) * 540) + (((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 1035) / 23) * 12)) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 1035))];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_1(float* __restrict__ T_transpose, float* __restrict__ ph) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 3105) {
    T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] / ph[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(540) default_function_kernel_5(float* __restrict__ T_softmax_maxelem, float* __restrict__ T_transpose) {
  T_softmax_maxelem[((int)threadIdx.x)] = 0.000000e+00f;
  for (int k = 0; k < 23; ++k) {
    T_softmax_maxelem[((int)threadIdx.x)] = (T_softmax_maxelem[((int)threadIdx.x)] + T_transpose[((((int)threadIdx.x) * 23) + k)]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_3(float* __restrict__ T_softmax_maxelem, float* __restrict__ T_transpose) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 3105) {
    T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] - T_softmax_maxelem[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 23)]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_4(float* __restrict__ T_transpose) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 3105) {
      int v_ = ((int)(floorf(((max(min(T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) + 1.270000e+02f)) << 23;
    T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max(((*(float *)(&(v_))) * ((((((((((((((1.987569e-04f * (max(min(T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.398200e-03f) * (max(min(T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 8.333452e-03f) * (max(min(T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 4.166580e-02f) * (max(min(T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.666667e-01f) * (max(min(T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 5.000000e-01f) * (max(min(T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) * (max(min(T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + (max(min(T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.000000e+00f)), T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(540) default_function_kernel_2(float* __restrict__ T_softmax_maxelem, float* __restrict__ T_transpose) {
  T_softmax_maxelem[((int)threadIdx.x)] = -3.402823e+38f;
  for (int k = 0; k < 23; ++k) {
    T_softmax_maxelem[((int)threadIdx.x)] = max(T_softmax_maxelem[((int)threadIdx.x)], T_transpose[((((int)threadIdx.x) * 23) + k)]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_6(float* __restrict__ T_softmax_maxelem, float* __restrict__ T_softmax_norm, float* __restrict__ T_transpose) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 3105) {
    T_softmax_norm[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] / T_softmax_maxelem[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 23)]);
  }
}

