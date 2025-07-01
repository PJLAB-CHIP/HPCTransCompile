extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_2(float* __restrict__ T_reshape, float* __restrict__ T_softmax_maxelem) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 2541) {
    T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] - T_softmax_maxelem[(((((int)blockIdx.x) * 512) + (((int)threadIdx.x) >> 1)) / 21)]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_6(float* __restrict__ T_broadcast_to, float* __restrict__ T_reshape) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 17787) {
    T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_reshape[((((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 1617) * 924) + (((((((int)blockIdx.x) * 50) + (((int)threadIdx.x) >> 1)) % 462) / 21) * 42)) + (((((int)blockIdx.x) * 16) + ((int)threadIdx.x)) % 42))];
  }
}

extern "C" __global__ void __launch_bounds__(242) default_function_kernel_4(float* __restrict__ T_reshape, float* __restrict__ T_softmax_maxelem) {
  T_softmax_maxelem[((int)threadIdx.x)] = 0.000000e+00f;
  for (int k = 0; k < 42; ++k) {
    T_softmax_maxelem[((int)threadIdx.x)] = (T_softmax_maxelem[((int)threadIdx.x)] + T_reshape[((((int)threadIdx.x) * 42) + k)]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_7(float* __restrict__ T_broadcast_to, float* __restrict__ T_reverse_sequence) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 17787) {
    T_reverse_sequence[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_broadcast_to[(((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 6468) + 64680) - ((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 1617) * 6468))];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_5(float* __restrict__ T_reshape, float* __restrict__ T_softmax_maxelem) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 2541) {
    T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] / T_softmax_maxelem[(((((int)blockIdx.x) * 512) + (((int)threadIdx.x) >> 1)) / 21)]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_3(float* __restrict__ T_reshape) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 2541) {
      int v_ = ((int)(floorf(((max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) + 1.270000e+02f)) << 23;
    T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max(((*(float *)(&(v_))) * ((((((((((((((1.987569e-04f * (max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.398200e-03f) * (max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 8.333452e-03f) * (max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 4.166580e-02f) * (max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.666667e-01f) * (max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 5.000000e-01f) * (max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) * (max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + (max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.000000e+00f)), T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ T_reshape, float* __restrict__ ph) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 2541) {
    T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ph[(((((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 231) * 924) + (((((((int)blockIdx.x) * 50) + (((int)threadIdx.x) >> 1)) % 462) / 21) * 42)) + (((((((int)blockIdx.x) * 16) + ((int)threadIdx.x)) % 42) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7))];
  }
}

extern "C" __global__ void __launch_bounds__(242) default_function_kernel_1(float* __restrict__ T_reshape, float* __restrict__ T_softmax_maxelem) {
  T_softmax_maxelem[((int)threadIdx.x)] = -3.402823e+38f;
  for (int k = 0; k < 42; ++k) {
    T_softmax_maxelem[((int)threadIdx.x)] = max(T_softmax_maxelem[((int)threadIdx.x)], T_reshape[((((int)threadIdx.x) * 42) + k)]);
  }
}

