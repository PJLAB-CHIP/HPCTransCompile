extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_2(float* __restrict__ T_reshape) {
  if (((((int)blockIdx.x) * 32) + (((int)threadIdx.x) >> 5)) < 45) {
      int v_ = ((int)(floorf(((max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) + 1.270000e+02f)) << 23;
    T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max(((*(float *)(&(v_))) * ((((((((((((((1.987569e-04f * (max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.398200e-03f) * (max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 8.333452e-03f) * (max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 4.166580e-02f) * (max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.666667e-01f) * (max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 5.000000e-01f) * (max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) * (max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + (max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.000000e+00f)), T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_3(float* __restrict__ T_reshape, float* __restrict__ T_reshape_1) {
  if (((((int)blockIdx.x) * 32) + (((int)threadIdx.x) >> 5)) < 45) {
    T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_reshape_1[((((((((((int)blockIdx.x) * 40) + (((int)threadIdx.x) >> 2)) % 72) / 3) * 60) + ((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) >> 5)) / 9) * 12)) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 6) * 2)) + ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 1)) % 6) / 3))];
  }
}

extern "C" __global__ void __launch_bounds__(60) default_function_kernel_4(float* __restrict__ T_reshape, float* __restrict__ T_transpose_red) {
  T_transpose_red[((int)threadIdx.x)] = 0.000000e+00f;
  for (int k1 = 0; k1 < 24; ++k1) {
    T_transpose_red[((int)threadIdx.x)] = (T_transpose_red[((int)threadIdx.x)] + T_reshape[((((((int)threadIdx.x) / 12) * 288) + (k1 * 12)) + (((int)threadIdx.x) % 12))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_1(float* __restrict__ T_reshape, float* __restrict__ T_reshape_1) {
  if (((((int)blockIdx.x) * 32) + (((int)threadIdx.x) >> 5)) < 45) {
    T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_reshape_1[((((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 15) * 60) + ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 1)) % 30) * 2)) + (((int)threadIdx.x) & 1))];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ T_reshape, float* __restrict__ ph) {
  if (((((int)blockIdx.x) * 32) + (((int)threadIdx.x) >> 5)) < 45) {
    T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ph[(((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) / 15) * 240) + (((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 2)) % 60) / 3) * 12)) + ((((((int)threadIdx.x) >> 2) + ((int)blockIdx.x)) % 3) * 4)) + (((int)threadIdx.x) & 3))];
  }
}

