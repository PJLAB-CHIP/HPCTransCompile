extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_2(float* __restrict__ T_broadcast_to) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 11979) {
    T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = sqrtf(T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ T_broadcast_to, float* __restrict__ ph) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 3993) {
    T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ph[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 1452)];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_4(float* __restrict__ T_broadcast_to, float* __restrict__ T_broadcast_to_1) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 11979) {
    T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] - T_broadcast_to_1[(((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) % 1089) / 33) * 528) + (((((((int)blockIdx.x) * 124) + (((int)threadIdx.x) >> 2)) % 132) / 11) * 44)) + (((((int)blockIdx.x) * 12) + ((int)threadIdx.x)) % 44))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_6(float* __restrict__ T_broadcast_to, float* __restrict__ T_broadcast_to_1) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 1089) {
    T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = 0.000000e+00f;
  }
  for (int k = 0; k < 11; ++k) {
    if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 1089) {
      T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + T_broadcast_to_1[(((k * 17424) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))]);
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_1(float* __restrict__ T_broadcast_to, float* __restrict__ T_broadcast_to_1) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 11979) {
    T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_broadcast_to_1[(((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) / 33) * 44) + (((((int)blockIdx.x) * 12) + ((int)threadIdx.x)) % 44))];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_5(float* __restrict__ T_broadcast_to) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 11979) {
      int v_ = ((int)(floorf(((max(min(T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) + 1.270000e+02f)) << 23;
    T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max(((*(float *)(&(v_))) * ((((((((((((((1.987569e-04f * (max(min(T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.398200e-03f) * (max(min(T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 8.333452e-03f) * (max(min(T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 4.166580e-02f) * (max(min(T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.666667e-01f) * (max(min(T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 5.000000e-01f) * (max(min(T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) * (max(min(T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + (max(min(T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.000000e+00f)), T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_7(float* __restrict__ T_broadcast_to, float* __restrict__ T_broadcast_to_1, float* __restrict__ T_softmax_norm) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 11979) {
    T_softmax_norm[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] / T_broadcast_to_1[(((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) % 1089) / 33) * 528) + (((((((int)blockIdx.x) * 124) + (((int)threadIdx.x) >> 2)) % 132) / 11) * 44)) + (((((int)blockIdx.x) * 12) + ((int)threadIdx.x)) % 44))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_3(float* __restrict__ T_broadcast_to, float* __restrict__ T_broadcast_to_1) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 1089) {
    T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = -3.402823e+38f;
  }
  for (int k = 0; k < 11; ++k) {
    if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 1089) {
      T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max(T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], T_broadcast_to_1[(((k * 17424) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))]);
    }
  }
}

