extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_5(float* __restrict__ T_expand_dims, float* __restrict__ T_reverse_sequence, float* __restrict__ compute) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 3795) {
    T_reverse_sequence[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ((T_reverse_sequence[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] - T_expand_dims[(((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 3)]) - __logf(compute[(((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 3)]));
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ T_reverse_sequence, float* __restrict__ ph) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 345) {
    T_reverse_sequence[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ph[(((((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 15) * 60) + (((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 12)) + 48) - (((((((int)threadIdx.x) >> 2) + ((int)blockIdx.x)) % 15) / 3) * 12))];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_2(float* __restrict__ T_expand_dims, float* __restrict__ T_reverse_sequence) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 3795) {
    T_reverse_sequence[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_expand_dims[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 1380)];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_1(float* __restrict__ T_expand_dims, float* __restrict__ T_reverse_sequence) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 345) {
    T_expand_dims[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_reverse_sequence[(((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 15) * 60) + (((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 60))];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_6(float* __restrict__ T_fast_exp, float* __restrict__ T_reverse_sequence) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 3795) {
      int v_ = ((int)(floorf(((max(min(T_reverse_sequence[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) + 1.270000e+02f)) << 23;
    T_fast_exp[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max(((*(float *)(&(v_))) * ((((((((((((((1.987569e-04f * (max(min(T_reverse_sequence[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reverse_sequence[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.398200e-03f) * (max(min(T_reverse_sequence[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reverse_sequence[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 8.333452e-03f) * (max(min(T_reverse_sequence[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reverse_sequence[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 4.166580e-02f) * (max(min(T_reverse_sequence[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reverse_sequence[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.666667e-01f) * (max(min(T_reverse_sequence[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reverse_sequence[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 5.000000e-01f) * (max(min(T_reverse_sequence[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reverse_sequence[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) * (max(min(T_reverse_sequence[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reverse_sequence[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + (max(min(T_reverse_sequence[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(T_reverse_sequence[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.000000e+00f)), T_reverse_sequence[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_4(float* __restrict__ T_expand_dims, float* __restrict__ T_reverse_sequence, float* __restrict__ compute) {
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 1265) {
    compute[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = 0.000000e+00f;
  }
  for (int k = 0; k < 12; ++k) {
    if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 1265) {
      compute[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (compute[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + __expf((T_reverse_sequence[(((((int)blockIdx.x) * 12288) + (((int)threadIdx.x) * 12)) + k)] - T_expand_dims[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))])));
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_3(float* __restrict__ T_expand_dims, float* __restrict__ T_reverse_sequence) {
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 1265) {
    T_expand_dims[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = -3.402823e+38f;
  }
  for (int k = 0; k < 12; ++k) {
    if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 1265) {
      T_expand_dims[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max(T_expand_dims[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], T_reverse_sequence[(((((int)blockIdx.x) * 12288) + (((int)threadIdx.x) * 12)) + k)]);
    }
  }
}

