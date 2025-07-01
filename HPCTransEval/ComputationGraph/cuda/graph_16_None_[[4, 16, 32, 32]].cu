extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_5(float* __restrict__ T_softmax_maxelem, float* __restrict__ resize) {
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) < 625) {
    resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] - T_softmax_maxelem[(((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) / 625) * 2500) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 2500))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ ph, float* __restrict__ resize) {
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) < 625) {
    resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ((((ph[((((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 625) * 1024) + (max(min(((int)floorf((((((float)((((((int)blockIdx.x) * 512) + (((int)threadIdx.x) >> 1)) % 1250) / 25)) + 5.000000e-01f) * 6.400000e-01f) - 5.000000e-01f))), 31), 0) * 32)) + max(min(((int)floorf((((((float)(((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) % 50)) + 5.000000e-01f) * 6.400000e-01f) - 5.000000e-01f))), 31), 0))] * (1.000000e+00f - ((((((float)(((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) % 50)) + 5.000000e-01f) * 6.400000e-01f) - 5.000000e-01f) - ((float)((int)floorf((((((float)(((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) % 50)) + 5.000000e-01f) * 6.400000e-01f) - 5.000000e-01f))))))) + (ph[((((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 625) * 1024) + (max(min(((int)floorf((((((float)((((((int)blockIdx.x) * 512) + (((int)threadIdx.x) >> 1)) % 1250) / 25)) + 5.000000e-01f) * 6.400000e-01f) - 5.000000e-01f))), 31), 0) * 32)) + max(min((((int)floorf((((((float)(((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) % 50)) + 5.000000e-01f) * 6.400000e-01f) - 5.000000e-01f))) + 1), 31), 0))] * ((((((float)(((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) % 50)) + 5.000000e-01f) * 6.400000e-01f) - 5.000000e-01f) - ((float)((int)floorf((((((float)(((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) % 50)) + 5.000000e-01f) * 6.400000e-01f) - 5.000000e-01f))))))) * (1.000000e+00f - ((((((float)((((((int)blockIdx.x) * 512) + (((int)threadIdx.x) >> 1)) % 1250) / 25)) + 5.000000e-01f) * 6.400000e-01f) - 5.000000e-01f) - ((float)((int)floorf((((((float)((((((int)blockIdx.x) * 512) + (((int)threadIdx.x) >> 1)) % 1250) / 25)) + 5.000000e-01f) * 6.400000e-01f) - 5.000000e-01f))))))) + (((ph[((((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 625) * 1024) + (max(min((((int)floorf((((((float)((((((int)blockIdx.x) * 512) + (((int)threadIdx.x) >> 1)) % 1250) / 25)) + 5.000000e-01f) * 6.400000e-01f) - 5.000000e-01f))) + 1), 31), 0) * 32)) + max(min(((int)floorf((((((float)(((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) % 50)) + 5.000000e-01f) * 6.400000e-01f) - 5.000000e-01f))), 31), 0))] * (1.000000e+00f - ((((((float)(((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) % 50)) + 5.000000e-01f) * 6.400000e-01f) - 5.000000e-01f) - ((float)((int)floorf((((((float)(((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) % 50)) + 5.000000e-01f) * 6.400000e-01f) - 5.000000e-01f))))))) + (ph[((((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 625) * 1024) + (max(min((((int)floorf((((((float)((((((int)blockIdx.x) * 512) + (((int)threadIdx.x) >> 1)) % 1250) / 25)) + 5.000000e-01f) * 6.400000e-01f) - 5.000000e-01f))) + 1), 31), 0) * 32)) + max(min((((int)floorf((((((float)(((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) % 50)) + 5.000000e-01f) * 6.400000e-01f) - 5.000000e-01f))) + 1), 31), 0))] * ((((((float)(((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) % 50)) + 5.000000e-01f) * 6.400000e-01f) - 5.000000e-01f) - ((float)((int)floorf((((((float)(((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) % 50)) + 5.000000e-01f) * 6.400000e-01f) - 5.000000e-01f))))))) * ((((((float)((((((int)blockIdx.x) * 512) + (((int)threadIdx.x) >> 1)) % 1250) / 25)) + 5.000000e-01f) * 6.400000e-01f) - 5.000000e-01f) - ((float)((int)floorf((((((float)((((((int)blockIdx.x) * 512) + (((int)threadIdx.x) >> 1)) % 1250) / 25)) + 5.000000e-01f) * 6.400000e-01f) - 5.000000e-01f)))))));
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_6(float* __restrict__ resize) {
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) < 625) {
      int v_ = ((int)(floorf(((max(min(resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) + 1.270000e+02f)) << 23;
    resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max(((*(float *)(&(v_))) * ((((((((((((((1.987569e-04f * (max(min(resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.398200e-03f) * (max(min(resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 8.333452e-03f) * (max(min(resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 4.166580e-02f) * (max(min(resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.666667e-01f) * (max(min(resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 5.000000e-01f) * (max(min(resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) * (max(min(resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + (max(min(resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.000000e+00f)), resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_8(float* __restrict__ T_softmax_maxelem, float* __restrict__ T_softmax_norm, float* __restrict__ resize) {
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) < 625) {
    T_softmax_norm[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] / T_softmax_maxelem[(((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) / 625) * 2500) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 2500))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_3(float* __restrict__ resize) {
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) < 625) {
    resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ((0.000000e+00f < resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) ? resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] : (resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * 5.000000e-01f));
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_4(float* __restrict__ T_softmax_maxelem, float* __restrict__ resize) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 625) {
    T_softmax_maxelem[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = -3.402823e+38f;
  }
  for (int k = 0; k < 16; ++k) {
    if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 625) {
      T_softmax_maxelem[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max(T_softmax_maxelem[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], resize[((((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 625) * 40000) + (k * 2500)) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 2500))]);
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_7(float* __restrict__ T_softmax_maxelem, float* __restrict__ resize) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 625) {
    T_softmax_maxelem[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = 0.000000e+00f;
  }
  for (int k = 0; k < 16; ++k) {
    if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 625) {
      T_softmax_maxelem[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_softmax_maxelem[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + resize[((((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 625) * 40000) + (k * 2500)) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 2500))]);
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_2(float* __restrict__ resize) {
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) < 625) {
    resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * 7.692308e-02f);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_1(float* __restrict__ resize) {
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) < 625) {
    resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (resize[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + 3.000000e+00f);
  }
}

