extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_13(float* __restrict__ T_matmul_NN, float* __restrict__ T_transpose) {
  T_matmul_NN[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_matmul_NN[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_37(float* __restrict__ T_strided_slice, float* __restrict__ T_transpose) {
  T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))];
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_22(float* __restrict__ T_matmul_NN, float* __restrict__ T_strided_slice, float* __restrict__ T_transpose) {
  T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = 0.000000e+00f;
  for (int h = 0; h < 64; ++h) {
    T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + (T_matmul_NN[(((((int)blockIdx.x) * 512) + ((((int)threadIdx.x) >> 7) * 64)) + h)] * T_transpose[((((((int)blockIdx.x) >> 4) * 8192) + ((((int)threadIdx.x) & 127) * 64)) + h)]));
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_39(float* __restrict__ T_multiply, float* __restrict__ T_strided_slice) {
  if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 9)) < 129) {
    T_multiply[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (1.000000e+00f / (1.000000e+00f + __expf((0.000000e+00f - T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]))));
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_28(float* __restrict__ T_matmul_NN, float* __restrict__ T_transpose) {
  T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_matmul_NN[(((((((int)blockIdx.x) & 7) * 8192) + ((((int)threadIdx.x) >> 6) * 512)) + ((((int)blockIdx.x) >> 3) * 64)) + (((int)threadIdx.x) & 63))];
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_8(float* __restrict__ T_matmul_NN, float* __restrict__ T_transpose, float* __restrict__ ph) {
  T_matmul_NN[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * ph[(((((int)blockIdx.x) & 7) * 1024) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ T_cast, float* __restrict__ ph) {
  T_cast[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ph[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))];
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_15(float* __restrict__ T_strided_slice, float* __restrict__ T_transpose) {
  T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_transpose[(((((((int)blockIdx.x) & 7) * 8192) + ((((int)threadIdx.x) >> 6) * 512)) + ((((int)blockIdx.x) >> 3) * 64)) + (((int)threadIdx.x) & 63))];
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_40(float* __restrict__ T_multiply, float* __restrict__ T_strided_slice) {
  if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 9)) < 129) {
    T_multiply[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_multiply[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_2(float* __restrict__ T_cast, float* __restrict__ T_multiply) {
  T_multiply[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_cast[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_cast[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_11(float* __restrict__ T_strided_slice, float* __restrict__ T_strided_slice_1, float* __restrict__ T_transpose) {
  T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ((32 <= (((int)threadIdx.x) & 63)) ? T_strided_slice[((((((int)blockIdx.x) * 512) + ((((int)threadIdx.x) >> 6) * 32)) + (((int)threadIdx.x) & 63)) - 32)] : T_strided_slice_1[(((((int)blockIdx.x) * 512) + ((((int)threadIdx.x) >> 6) * 32)) + (((int)threadIdx.x) & 63))]);
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_19(float* __restrict__ T_strided_slice, float* __restrict__ T_strided_slice_1, float* __restrict__ T_strided_slice_2) {
  T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ((32 <= (((int)threadIdx.x) & 63)) ? T_strided_slice_1[((((((int)blockIdx.x) * 512) + ((((int)threadIdx.x) >> 6) * 32)) + (((int)threadIdx.x) & 63)) - 32)] : T_strided_slice_2[(((((int)blockIdx.x) * 512) + ((((int)threadIdx.x) >> 6) * 32)) + (((int)threadIdx.x) & 63))]);
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_23(float* __restrict__ T_cast, float* __restrict__ T_strided_slice) {
  T_cast[((int)threadIdx.x)] = -3.402823e+38f;
  for (int k = 0; k < 128; ++k) {
    T_cast[((int)threadIdx.x)] = max(T_cast[((int)threadIdx.x)], T_strided_slice[((((int)threadIdx.x) * 128) + k)]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_9(float* __restrict__ T_strided_slice, float* __restrict__ T_transpose) {
  T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_transpose[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 32)];
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_16(float* __restrict__ T_strided_slice, float* __restrict__ T_transpose, float* __restrict__ ph) {
  T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * ph[(((((int)blockIdx.x) & 7) * 1024) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_4(float* __restrict__ T_cast, float* __restrict__ T_cast_1, float* __restrict__ T_multiply_red) {
  T_cast[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ((T_cast[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_cast_1[(((int)threadIdx.x) & 511)]) * (1.000000e+00f / sqrtf(((T_multiply_red[((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 9))] * 1.953125e-03f) + 1.000000e-05f))));
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_25(float* __restrict__ T_cast, float* __restrict__ T_strided_slice) {
  T_cast[((int)threadIdx.x)] = 0.000000e+00f;
  for (int k = 0; k < 128; ++k) {
    T_cast[((int)threadIdx.x)] = (T_cast[((int)threadIdx.x)] + T_strided_slice[((((int)threadIdx.x) * 128) + k)]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_45(float* __restrict__ T_add, float* __restrict__ T_matmul_NN, float* __restrict__ T_transpose) {
  T_add[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + T_matmul_NN[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_7(float* __restrict__ T_matmul_NN, float* __restrict__ T_transpose) {
  T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_matmul_NN[(((((((int)blockIdx.x) & 7) * 8192) + ((((int)threadIdx.x) >> 6) * 512)) + ((((int)blockIdx.x) >> 3) * 64)) + (((int)threadIdx.x) & 63))];
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_6(float* __restrict__ T_matmul_NN, float* __restrict__ T_multiply, float* __restrict__ ph) {
  T_matmul_NN[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = 0.000000e+00f;
  for (int k = 0; k < 512; ++k) {
    T_matmul_NN[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_matmul_NN[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + (T_multiply[(((((int)blockIdx.x) * 1024) + ((((int)threadIdx.x) >> 9) * 512)) + k)] * ph[((k * 512) + (((int)threadIdx.x) & 511))]));
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_32(float* __restrict__ T_cast, float* __restrict__ T_transpose) {
  T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + T_cast[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(512) default_function_kernel_33(float* __restrict__ T_cast, float* __restrict__ ph) {
  T_cast[((int)threadIdx.x)] = ph[((int)threadIdx.x)];
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_20(float* __restrict__ T_strided_slice, float* __restrict__ ph) {
  T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * ph[(((((int)blockIdx.x) & 7) * 1024) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_12(float* __restrict__ T_transpose, float* __restrict__ ph) {
  T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * ph[(((((int)blockIdx.x) & 7) * 1024) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(128) default_function_kernel_3(float* __restrict__ T_multiply, float* __restrict__ T_multiply_red) {
  T_multiply_red[((int)threadIdx.x)] = 0.000000e+00f;
  for (int k2 = 0; k2 < 512; ++k2) {
    T_multiply_red[((int)threadIdx.x)] = (T_multiply_red[((int)threadIdx.x)] + T_multiply[((((int)threadIdx.x) * 512) + k2)]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_10(float* __restrict__ T_strided_slice, float* __restrict__ T_transpose) {
  T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_transpose[(((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31))];
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_5(float* __restrict__ T_cast, float* __restrict__ T_multiply) {
  T_multiply[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_cast[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))];
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_24(float* __restrict__ T_cast, float* __restrict__ T_strided_slice) {
  T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = __expf((T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] - T_cast[((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 7))]));
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_30(float* __restrict__ T_matmul_NN, float* __restrict__ T_multiply) {
  T_matmul_NN[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_multiply[((((((((int)threadIdx.x) & 511) >> 6) * 8192) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 9) * 64)) + (((int)threadIdx.x) & 63))];
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_43(float* __restrict__ T_multiply, float* __restrict__ T_strided_slice, float* __restrict__ ph) {
  if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 9)) < 129) {
    T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = 0.000000e+00f;
  }
  for (int k = 0; k < 1376; ++k) {
    if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 9)) < 129) {
      T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + (T_multiply[(((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) >> 5)) / 43) * 1376) + k)] * ph[((k * 1376) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 1376))]));
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_17(float* __restrict__ T_strided_slice, float* __restrict__ T_strided_slice_1) {
  T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_strided_slice_1[((((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31)) + 32)];
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_29(float* __restrict__ T_multiply, float* __restrict__ T_strided_slice, float* __restrict__ T_transpose) {
  T_multiply[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = 0.000000e+00f;
  for (int k = 0; k < 128; ++k) {
    T_multiply[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_multiply[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + (T_strided_slice[(((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 6) * 128)) + k)] * T_transpose[((((((int)blockIdx.x) >> 3) * 8192) + (k * 64)) + (((int)threadIdx.x) & 63))]));
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_18(float* __restrict__ T_strided_slice, float* __restrict__ T_strided_slice_1) {
  T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_strided_slice_1[(((((int)blockIdx.x) * 2048) + ((((int)threadIdx.x) >> 5) * 64)) + (((int)threadIdx.x) & 31))];
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_31(float* __restrict__ T_matmul_NN, float* __restrict__ T_transpose, float* __restrict__ ph) {
  T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = 0.000000e+00f;
  for (int k = 0; k < 512; ++k) {
    T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + (T_matmul_NN[(((((int)blockIdx.x) * 1024) + ((((int)threadIdx.x) >> 9) * 512)) + k)] * ph[((k * 512) + (((int)threadIdx.x) & 511))]));
  }
}

extern "C" __global__ void __launch_bounds__(512) default_function_kernel_1(float* __restrict__ T_cast, float* __restrict__ ph) {
  T_cast[((int)threadIdx.x)] = ph[((int)threadIdx.x)];
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_14(float* __restrict__ T_multiply, float* __restrict__ T_transpose, float* __restrict__ ph) {
  T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = 0.000000e+00f;
  for (int k = 0; k < 512; ++k) {
    T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + (T_multiply[(((((int)blockIdx.x) * 1024) + ((((int)threadIdx.x) >> 9) * 512)) + k)] * ph[((k * 512) + (((int)threadIdx.x) & 511))]));
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_21(float* __restrict__ T_strided_slice, float* __restrict__ T_transpose) {
  T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(128) default_function_kernel_35(float* __restrict__ T_multiply, float* __restrict__ T_multiply_red) {
  T_multiply_red[((int)threadIdx.x)] = 0.000000e+00f;
  for (int k2 = 0; k2 < 512; ++k2) {
    T_multiply_red[((int)threadIdx.x)] = (T_multiply_red[((int)threadIdx.x)] + T_multiply[((((int)threadIdx.x) * 512) + k2)]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_27(float* __restrict__ T_matmul_NN, float* __restrict__ T_multiply, float* __restrict__ ph) {
  T_matmul_NN[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = 0.000000e+00f;
  for (int k = 0; k < 512; ++k) {
    T_matmul_NN[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_matmul_NN[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + (T_multiply[(((((int)blockIdx.x) * 1024) + ((((int)threadIdx.x) >> 9) * 512)) + k)] * ph[((k * 512) + (((int)threadIdx.x) & 511))]));
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_34(float* __restrict__ T_multiply, float* __restrict__ T_transpose) {
  T_multiply[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_26(float* __restrict__ T_cast, float* __restrict__ T_strided_slice) {
  T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] / T_cast[((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 7))]);
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_44(float* __restrict__ T_matmul_NN, float* __restrict__ T_strided_slice) {
  T_matmul_NN[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))];
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_41(float* __restrict__ T_strided_slice, float* __restrict__ T_strided_slice_1, float* __restrict__ ph) {
  if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 9)) < 129) {
    T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = 0.000000e+00f;
  }
  for (int k = 0; k < 512; ++k) {
    if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 9)) < 129) {
      T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + (T_strided_slice_1[(((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) >> 5)) / 43) * 512) + k)] * ph[((k * 1376) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 1376))]));
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_38(float* __restrict__ T_strided_slice, float* __restrict__ T_strided_slice_1, float* __restrict__ ph) {
  if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 9)) < 129) {
    T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = 0.000000e+00f;
  }
  for (int k = 0; k < 512; ++k) {
    if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 9)) < 129) {
      T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + (T_strided_slice_1[(((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) >> 5)) / 43) * 512) + k)] * ph[((k * 1376) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 1376))]));
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_42(float* __restrict__ T_multiply, float* __restrict__ T_strided_slice) {
  if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 9)) < 129) {
    T_multiply[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_multiply[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_36(float* __restrict__ T_cast, float* __restrict__ T_multiply_red, float* __restrict__ T_transpose) {
  T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ((T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_cast[(((int)threadIdx.x) & 511)]) * (1.000000e+00f / sqrtf(((T_multiply_red[((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 9))] * 1.953125e-03f) + 1.000000e-05f))));
}

