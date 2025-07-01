extern "C" __global__ void __launch_bounds__(3) default_function_kernel_4(float* __restrict__ T_reshape, float* __restrict__ ph) {
  T_reshape[((int)threadIdx.x)] = ph[((int)threadIdx.x)];
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_15(float* __restrict__ T_reshape, float* __restrict__ group_conv2d_nchw, float* __restrict__ pad_temp) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 11907) {
    group_conv2d_nchw[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (pad_temp[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] - T_reshape[(((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 3969)]);
  }
}

extern "C" __global__ void __launch_bounds__(3) default_function_kernel_18(float* __restrict__ T_reshape) {
  T_reshape[((int)threadIdx.x)] = sqrtf(T_reshape[((int)threadIdx.x)]);
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_23(float* __restrict__ T_reshape, float* __restrict__ group_conv2d_nchw) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 11907) {
    group_conv2d_nchw[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (group_conv2d_nchw[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + T_reshape[(((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 3969)]);
  }
}

extern "C" __global__ void __launch_bounds__(3) default_function_kernel_20(float* __restrict__ T_reshape, float* __restrict__ ph) {
  T_reshape[((int)threadIdx.x)] = ph[0];
}

extern "C" __global__ void __launch_bounds__(3) default_function_kernel_17(float* __restrict__ T_reshape) {
  T_reshape[((int)threadIdx.x)] = (T_reshape[((int)threadIdx.x)] + 1.000000e-05f);
}

extern "C" __global__ void __launch_bounds__(3) default_function_kernel_6(float* __restrict__ T_reshape) {
  T_reshape[((int)threadIdx.x)] = sqrtf(T_reshape[((int)threadIdx.x)]);
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_21(float* __restrict__ T_reshape, float* __restrict__ group_conv2d_nchw) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 11907) {
    group_conv2d_nchw[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (group_conv2d_nchw[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_reshape[(((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 3969)]);
  }
}

extern "C" __global__ void __launch_bounds__(3) default_function_kernel_22(float* __restrict__ T_reshape, float* __restrict__ ph) {
  T_reshape[((int)threadIdx.x)] = ph[0];
}

extern "C" __global__ void __launch_bounds__(3) default_function_kernel_2(float* __restrict__ T_reshape, float* __restrict__ ph) {
  T_reshape[((int)threadIdx.x)] = ph[((int)threadIdx.x)];
}

extern "C" __global__ void __launch_bounds__(3) default_function_kernel_8(float* __restrict__ T_reshape, float* __restrict__ ph) {
  T_reshape[((int)threadIdx.x)] = ph[0];
}

extern "C" __global__ void __launch_bounds__(3) default_function_kernel_5(float* __restrict__ T_reshape) {
  T_reshape[((int)threadIdx.x)] = (T_reshape[((int)threadIdx.x)] + 1.000000e-05f);
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_19(float* __restrict__ T_reshape, float* __restrict__ group_conv2d_nchw) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 11907) {
    group_conv2d_nchw[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (group_conv2d_nchw[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] / T_reshape[(((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 3969)]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_13(float* __restrict__ group_conv2d_nchw, float* __restrict__ pad_temp, float* __restrict__ ph) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 11907) {
    pad_temp[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = 0.000000e+00f;
  }
  for (int rc = 0; rc < 3; ++rc) {
    if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 11907) {
      pad_temp[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (pad_temp[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + (group_conv2d_nchw[((rc * 15876) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 15876))] * ph[(((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 3969) * 9) + rc)]));
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_9(float* __restrict__ T_reshape, float* __restrict__ group_conv2d_nchw) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 11907) {
    group_conv2d_nchw[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (group_conv2d_nchw[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_reshape[(((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 3969)]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_7(float* __restrict__ T_reshape, float* __restrict__ group_conv2d_nchw) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 11907) {
    group_conv2d_nchw[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (group_conv2d_nchw[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] / T_reshape[(((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 3969)]);
  }
}

extern "C" __global__ void __launch_bounds__(3) default_function_kernel_10(float* __restrict__ T_reshape, float* __restrict__ ph) {
  T_reshape[((int)threadIdx.x)] = ph[0];
}

extern "C" __global__ void __launch_bounds__(3) default_function_kernel_14(float* __restrict__ T_reshape, float* __restrict__ ph) {
  T_reshape[((int)threadIdx.x)] = ph[((int)threadIdx.x)];
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_12(float* __restrict__ group_conv2d_nchw) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 11907) {
    group_conv2d_nchw[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max(group_conv2d_nchw[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 0.000000e+00f);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_11(float* __restrict__ T_reshape, float* __restrict__ group_conv2d_nchw) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 11907) {
    group_conv2d_nchw[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (group_conv2d_nchw[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + T_reshape[(((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 3969)]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ pad_temp, float* __restrict__ ph) {
  pad_temp[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ph[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))];
}

extern "C" __global__ void __launch_bounds__(3) default_function_kernel_16(float* __restrict__ T_reshape, float* __restrict__ ph) {
  T_reshape[((int)threadIdx.x)] = ph[((int)threadIdx.x)];
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_24(float* __restrict__ compute, float* __restrict__ group_conv2d_nchw) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 11907) {
    compute[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max(group_conv2d_nchw[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 0.000000e+00f);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_1(float* __restrict__ group_conv2d_nchw, float* __restrict__ pad_temp, float* __restrict__ ph) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 11907) {
    group_conv2d_nchw[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = 0.000000e+00f;
  }
  for (int ry = 0; ry < 3; ++ry) {
    for (int rx = 0; rx < 3; ++rx) {
      if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 11907) {
        group_conv2d_nchw[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (group_conv2d_nchw[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + (pad_temp[((((((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 3969) * 16384) + (((((((int)blockIdx.x) * 512) + (((int)threadIdx.x) >> 1)) % 7938) / 63) * 128)) + (ry * 128)) + rx) + (((((int)blockIdx.x) * 16) + ((int)threadIdx.x)) % 126))] * ph[((((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 3969) * 27) + (ry * 3)) + rx)]));
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_3(float* __restrict__ T_reshape, float* __restrict__ group_conv2d_nchw) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 11907) {
    group_conv2d_nchw[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (group_conv2d_nchw[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] - T_reshape[(((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 3969)]);
  }
}

