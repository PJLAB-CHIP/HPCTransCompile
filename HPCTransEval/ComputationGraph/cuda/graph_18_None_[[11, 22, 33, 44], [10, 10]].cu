extern "C" __global__ void __launch_bounds__(100) default_function_kernel_8(float* __restrict__ T_cast, float* __restrict__ ph) {
  T_cast[((int)threadIdx.x)] = ph[((int)threadIdx.x)];
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_9(float* __restrict__ T_concat, float* __restrict__ T_strided_slice) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 1125) {
    T_concat[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(180) default_function_kernel_10(float* __restrict__ T_concat, float* __restrict__ T_multiply_red) {
  T_multiply_red[((int)threadIdx.x)] = 0.000000e+00f;
  for (int k1 = 0; k1 < 10; ++k1) {
    for (int k2 = 0; k2 < 10; ++k2) {
      T_multiply_red[((int)threadIdx.x)] = (T_multiply_red[((int)threadIdx.x)] + T_concat[(((((((int)threadIdx.x) / 9) * 900) + (k1 * 90)) + (k2 * 9)) + (((int)threadIdx.x) % 9))]);
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_12(float* __restrict__ T_cast, float* __restrict__ T_strided_slice) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 1125) {
    T_cast[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_2(float* __restrict__ T_concat, float* __restrict__ T_strided_slice, float* __restrict__ T_strided_slice_1) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 1125) {
    T_concat[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ((1125 <= ((((int)blockIdx.x) * 128) + (((int)threadIdx.x) >> 3))) ? T_strided_slice[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) - 9000)] : T_strided_slice_1[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_6(float* __restrict__ T_concat, float* __restrict__ T_strided_slice) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 1125) {
    T_concat[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_concat[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] / T_strided_slice[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 9)]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_1(float* __restrict__ T_strided_slice, float* __restrict__ ph) {
  if (((((int)blockIdx.x) * 128) + (((int)threadIdx.x) >> 3)) < 1125) {
    T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ph[((((((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 225) * 31944) + (((((((int)blockIdx.x) * 62) + (((int)threadIdx.x) >> 1)) % 450) / 45) * 1452)) + (((((((int)blockIdx.x) * 34) + ((int)threadIdx.x)) % 90) / 9) * 44)) + (((((int)blockIdx.x) * 7) + ((int)threadIdx.x)) % 9)) + 16973)];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_7(float* __restrict__ T_concat, float* __restrict__ T_strided_slice) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 1125) {
    T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_concat[(((((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 225) * 900) + (((((((int)blockIdx.x) * 34) + ((int)threadIdx.x)) % 90) / 9) * 90)) + (((((((int)blockIdx.x) * 62) + (((int)threadIdx.x) >> 1)) % 450) / 45) * 9)) + (((((int)blockIdx.x) * 7) + ((int)threadIdx.x)) % 9))];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_5(float* __restrict__ T_concat, float* __restrict__ T_strided_slice) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 125) {
    T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = 0.000000e+00f;
  }
  for (int k = 0; k < 9; ++k) {
    if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 125) {
      T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + T_concat[(((((int)blockIdx.x) * 9216) + (((int)threadIdx.x) * 9)) + k)]);
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ T_strided_slice, float* __restrict__ ph) {
  if (((((int)blockIdx.x) * 128) + (((int)threadIdx.x) >> 3)) < 1125) {
    T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ph[((((((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 225) * 31944) + (((((((int)blockIdx.x) * 62) + (((int)threadIdx.x) >> 1)) % 450) / 45) * 1452)) + (((((((int)blockIdx.x) * 34) + ((int)threadIdx.x)) % 90) / 9) * 44)) + (((((int)blockIdx.x) * 7) + ((int)threadIdx.x)) % 9)) + 50414)];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_11(float* __restrict__ T_cast, float* __restrict__ T_multiply_red, float* __restrict__ T_strided_slice) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 1125) {
    T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ((T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_cast[((((((((int)blockIdx.x) * 62) + (((int)threadIdx.x) >> 1)) % 450) / 45) * 10) + ((((((int)blockIdx.x) * 34) + ((int)threadIdx.x)) % 90) / 9))]) * (1.000000e+00f / sqrtf(((T_multiply_red[(((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 225) * 9) + (((((int)blockIdx.x) * 7) + ((int)threadIdx.x)) % 9))] * 1.000000e-02f) + 1.000000e-05f))));
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_3(float* __restrict__ T_concat, float* __restrict__ T_strided_slice) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 125) {
    T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = -3.402823e+38f;
  }
  for (int k = 0; k < 9; ++k) {
    if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 125) {
      T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max(T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], T_concat[(((((int)blockIdx.x) * 9216) + (((int)threadIdx.x) * 9)) + k)]);
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_4(float* __restrict__ T_concat, float* __restrict__ T_strided_slice) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 1125) {
    T_concat[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = __expf((T_concat[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] - T_strided_slice[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 9)]));
  }
}

