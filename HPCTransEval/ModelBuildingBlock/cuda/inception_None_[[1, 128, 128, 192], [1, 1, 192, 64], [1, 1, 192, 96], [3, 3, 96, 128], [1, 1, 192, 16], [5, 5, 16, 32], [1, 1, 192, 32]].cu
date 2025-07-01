extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_1(float* __restrict__ pad_temp, float* __restrict__ pool_max) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 12; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    pool_max[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = -3.402823e+38f;
    for (int rv0 = 0; rv0 < 3; ++rv0) {
      for (int rv1 = 0; rv1 < 3; ++rv1) {
        pool_max[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = max(pool_max[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))], pad_temp[(((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 32) + (((int)blockIdx.x) >> 3)) / 3) * 24960) + (rv0 * 24960)) + (rv1 * 192)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 24576))]);
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_12(float* __restrict__ conv2d_nhwc) {
  for (int i0_i1_fused_i2_fused_i3_fused_outer = 0; i0_i1_fused_i2_fused_i3_fused_outer < 6; ++i0_i1_fused_i2_fused_i3_fused_outer) {
    conv2d_nhwc[(((i0_i1_fused_i2_fused_i3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = max(conv2d_nhwc[(((i0_i1_fused_i2_fused_i3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))], 0.000000e+00f);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_13(float* __restrict__ conv2d_nhwc, float* __restrict__ pad_temp) {
  for (int i0_i1_fused_i2_fused_i3_fused_outer = 0; i0_i1_fused_i2_fused_i3_fused_outer < 7; ++i0_i1_fused_i2_fused_i3_fused_outer) {
    if ((((i0_i1_fused_i2_fused_i3_fused_outer * 2048) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) >> 7)) < 12675) {
      pad_temp[(((i0_i1_fused_i2_fused_i3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = (((((195 <= (((i0_i1_fused_i2_fused_i3_fused_outer * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) >> 6))) && ((((i0_i1_fused_i2_fused_i3_fused_outer * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) >> 6)) < 25155)) && (3 <= ((((i0_i1_fused_i2_fused_i3_fused_outer * 8192) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) >> 5)) % 390))) && (((((i0_i1_fused_i2_fused_i3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 12480) < 12384)) ? conv2d_nhwc[(((((((i0_i1_fused_i2_fused_i3_fused_outer * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) >> 6)) / 195) * 12288) + ((((i0_i1_fused_i2_fused_i3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 12480)) - 12384)] : 0.000000e+00f);
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_4(float* __restrict__ pad_temp, float* __restrict__ ph) {
  for (int i0_i1_fused_i2_fused_i3_fused_outer = 0; i0_i1_fused_i2_fused_i3_fused_outer < 12; ++i0_i1_fused_i2_fused_i3_fused_outer) {
    pad_temp[(((i0_i1_fused_i2_fused_i3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = ph[(((i0_i1_fused_i2_fused_i3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_3(float* __restrict__ pad_temp, float* __restrict__ pool_max) {
  for (int i0_i1_fused_i2_fused_i3_fused_outer = 0; i0_i1_fused_i2_fused_i3_fused_outer < 2; ++i0_i1_fused_i2_fused_i3_fused_outer) {
    pool_max[(((i0_i1_fused_i2_fused_i3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = max(pad_temp[(((i0_i1_fused_i2_fused_i3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))], 0.000000e+00f);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_10(float* __restrict__ pad_temp, float* __restrict__ ph) {
  for (int i0_i1_fused_i2_fused_i3_fused_outer = 0; i0_i1_fused_i2_fused_i3_fused_outer < 12; ++i0_i1_fused_i2_fused_i3_fused_outer) {
    pad_temp[(((i0_i1_fused_i2_fused_i3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = ph[(((i0_i1_fused_i2_fused_i3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_7(float* __restrict__ conv2d_nhwc, float* __restrict__ pad_temp) {
  for (int i0_i1_fused_i2_fused_i3_fused_outer = 0; i0_i1_fused_i2_fused_i3_fused_outer < 2; ++i0_i1_fused_i2_fused_i3_fused_outer) {
    if ((((i0_i1_fused_i2_fused_i3_fused_outer * 1024) + (((int)blockIdx.x) * 4)) + (((int)threadIdx.x) >> 8)) < 1089) {
      pad_temp[(((i0_i1_fused_i2_fused_i3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = (((((33 <= (((i0_i1_fused_i2_fused_i3_fused_outer * 2048) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) >> 7))) && ((((i0_i1_fused_i2_fused_i3_fused_outer * 2048) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) >> 7)) < 2145)) && (1 <= ((((i0_i1_fused_i2_fused_i3_fused_outer * 8192) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) >> 5)) % 66))) && (((((i0_i1_fused_i2_fused_i3_fused_outer * 16384) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) >> 4)) % 132) < 130)) ? conv2d_nhwc[((((((((i0_i1_fused_i2_fused_i3_fused_outer * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) >> 6)) / 33) * 2048) + (((((i0_i1_fused_i2_fused_i3_fused_outer * 16384) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) >> 4)) % 132) * 16)) + (((int)threadIdx.x) & 15)) - 4128)] : 0.000000e+00f);
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_14(float* __restrict__ conv2d_nhwc, float* __restrict__ pad_temp, float* __restrict__ ph) {
  for (int nn_yy_fused_xx_fused_ff_fused_outer = 0; nn_yy_fused_xx_fused_ff_fused_outer < 8; ++nn_yy_fused_xx_fused_ff_fused_outer) {
    conv2d_nhwc[(((nn_yy_fused_xx_fused_ff_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = 0.000000e+00f;
    for (int ry = 0; ry < 3; ++ry) {
      for (int rx = 0; rx < 3; ++rx) {
        for (int rc = 0; rc < 96; ++rc) {
          conv2d_nhwc[(((nn_yy_fused_xx_fused_ff_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = (conv2d_nhwc[(((nn_yy_fused_xx_fused_ff_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] + (pad_temp[(((((((nn_yy_fused_xx_fused_ff_fused_outer * 199680) + ((((int)blockIdx.x) >> 4) * 12480)) + (ry * 12480)) + ((((int)blockIdx.x) & 15) * 768)) + ((((int)threadIdx.x) >> 7) * 96)) + (rx * 96)) + rc)] * ph[((((ry * 36864) + (rx * 12288)) + (rc * 128)) + (((int)threadIdx.x) & 127))]));
        }
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_19(float* __restrict__ T_concat, float* __restrict__ conv2d_nhwc, float* __restrict__ conv2d_nhwc_1, float* __restrict__ conv2d_nhwc_2, float* __restrict__ pool_max) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 16; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    T_concat[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = ((224 <= (((int)threadIdx.x) & 255)) ? pool_max[(((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 32768) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 8) * 32)) + (((int)threadIdx.x) & 255)) - 224)] : ((192 <= (((int)threadIdx.x) & 255)) ? conv2d_nhwc[(((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 32768) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 8) * 32)) + (((int)threadIdx.x) & 255)) - 192)] : ((64 <= (((int)threadIdx.x) & 255)) ? conv2d_nhwc_1[(((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 131072) + (((int)blockIdx.x) * 512)) + ((((int)threadIdx.x) >> 8) * 128)) + (((int)threadIdx.x) & 255)) - 64)] : conv2d_nhwc_2[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 65536) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.x) >> 8) * 64)) + (((int)threadIdx.x) & 255))])));
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_9(float* __restrict__ conv2d_nhwc) {
  for (int i0_i1_fused_i2_fused_i3_fused_outer = 0; i0_i1_fused_i2_fused_i3_fused_outer < 2; ++i0_i1_fused_i2_fused_i3_fused_outer) {
    conv2d_nhwc[(((i0_i1_fused_i2_fused_i3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = max(conv2d_nhwc[(((i0_i1_fused_i2_fused_i3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))], 0.000000e+00f);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_5(float* __restrict__ conv2d_nhwc, float* __restrict__ pad_temp, float* __restrict__ ph) {
  conv2d_nhwc[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = 0.000000e+00f;
  for (int rc = 0; rc < 192; ++rc) {
    conv2d_nhwc[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (conv2d_nhwc[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + (pad_temp[(((((int)blockIdx.x) * 12288) + ((((int)threadIdx.x) >> 4) * 192)) + rc)] * ph[((rc * 16) + (((int)threadIdx.x) & 15))]));
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_6(float* __restrict__ conv2d_nhwc) {
  conv2d_nhwc[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max(conv2d_nhwc[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 0.000000e+00f);
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_8(float* __restrict__ conv2d_nhwc, float* __restrict__ pad_temp, float* __restrict__ ph) {
  for (int nn_yy_fused_xx_fused_ff_fused_outer = 0; nn_yy_fused_xx_fused_ff_fused_outer < 2; ++nn_yy_fused_xx_fused_ff_fused_outer) {
    conv2d_nhwc[(((nn_yy_fused_xx_fused_ff_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = 0.000000e+00f;
    for (int ry = 0; ry < 5; ++ry) {
      for (int rx = 0; rx < 5; ++rx) {
        for (int rc = 0; rc < 16; ++rc) {
          conv2d_nhwc[(((nn_yy_fused_xx_fused_ff_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = (conv2d_nhwc[(((nn_yy_fused_xx_fused_ff_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] + (pad_temp[(((((((nn_yy_fused_xx_fused_ff_fused_outer * 135168) + ((((int)blockIdx.x) >> 2) * 2112)) + (ry * 2112)) + ((((int)blockIdx.x) & 3) * 512)) + ((((int)threadIdx.x) >> 5) * 16)) + (rx * 16)) + rc)] * ph[((((ry * 2560) + (rx * 512)) + (rc * 32)) + (((int)threadIdx.x) & 31))]));
        }
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_17(float* __restrict__ conv2d_nhwc, float* __restrict__ pad_temp, float* __restrict__ ph) {
  for (int nn_yy_fused_xx_fused_ff_fused_outer = 0; nn_yy_fused_xx_fused_ff_fused_outer < 4; ++nn_yy_fused_xx_fused_ff_fused_outer) {
    conv2d_nhwc[(((nn_yy_fused_xx_fused_ff_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = 0.000000e+00f;
    for (int rc = 0; rc < 192; ++rc) {
      conv2d_nhwc[(((nn_yy_fused_xx_fused_ff_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = (conv2d_nhwc[(((nn_yy_fused_xx_fused_ff_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] + (pad_temp[((((nn_yy_fused_xx_fused_ff_fused_outer * 786432) + (((int)blockIdx.x) * 3072)) + ((((int)threadIdx.x) >> 6) * 192)) + rc)] * ph[((rc * 64) + (((int)threadIdx.x) & 63))]));
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_16(float* __restrict__ pad_temp, float* __restrict__ ph) {
  for (int i0_i1_fused_i2_fused_i3_fused_outer = 0; i0_i1_fused_i2_fused_i3_fused_outer < 12; ++i0_i1_fused_i2_fused_i3_fused_outer) {
    pad_temp[(((i0_i1_fused_i2_fused_i3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = ph[(((i0_i1_fused_i2_fused_i3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_18(float* __restrict__ conv2d_nhwc) {
  for (int i0_i1_fused_i2_fused_i3_fused_outer = 0; i0_i1_fused_i2_fused_i3_fused_outer < 4; ++i0_i1_fused_i2_fused_i3_fused_outer) {
    conv2d_nhwc[(((i0_i1_fused_i2_fused_i3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = max(conv2d_nhwc[(((i0_i1_fused_i2_fused_i3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))], 0.000000e+00f);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_11(float* __restrict__ conv2d_nhwc, float* __restrict__ pad_temp, float* __restrict__ ph) {
  for (int nn_yy_fused_xx_fused_ff_fused_outer = 0; nn_yy_fused_xx_fused_ff_fused_outer < 6; ++nn_yy_fused_xx_fused_ff_fused_outer) {
    conv2d_nhwc[(((nn_yy_fused_xx_fused_ff_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = 0.000000e+00f;
    for (int rc = 0; rc < 192; ++rc) {
      conv2d_nhwc[(((nn_yy_fused_xx_fused_ff_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = (conv2d_nhwc[(((nn_yy_fused_xx_fused_ff_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] + (pad_temp[((((((nn_yy_fused_xx_fused_ff_fused_outer * 8192) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) >> 5)) / 3) * 192) + rc)] * ph[((rc * 96) + ((((nn_yy_fused_xx_fused_ff_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 96))]));
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_15(float* __restrict__ conv2d_nhwc) {
  for (int i0_i1_fused_i2_fused_i3_fused_outer = 0; i0_i1_fused_i2_fused_i3_fused_outer < 8; ++i0_i1_fused_i2_fused_i3_fused_outer) {
    conv2d_nhwc[(((i0_i1_fused_i2_fused_i3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = max(conv2d_nhwc[(((i0_i1_fused_i2_fused_i3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))], 0.000000e+00f);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ pad_temp, float* __restrict__ ph) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 13; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 1024) + (((int)blockIdx.x) * 4)) + (((int)threadIdx.x) >> 8)) < 12675) {
      pad_temp[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = (((((195 <= (((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 2048) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) >> 7))) && ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 2048) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) >> 7)) < 25155)) && (3 <= ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 4096) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) >> 6)) % 390))) && (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 24960) < 24768)) ? ph[(((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 2048) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) >> 7)) / 195) * 24576) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) % 24960)) - 24768)] : -3.402823e+38f);
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_2(float* __restrict__ pad_temp, float* __restrict__ ph, float* __restrict__ pool_max) {
  for (int nn_yy_fused_xx_fused_ff_fused_outer = 0; nn_yy_fused_xx_fused_ff_fused_outer < 2; ++nn_yy_fused_xx_fused_ff_fused_outer) {
    pad_temp[(((nn_yy_fused_xx_fused_ff_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = 0.000000e+00f;
    for (int rc = 0; rc < 192; ++rc) {
      pad_temp[(((nn_yy_fused_xx_fused_ff_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] = (pad_temp[(((nn_yy_fused_xx_fused_ff_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x))] + (pool_max[((((nn_yy_fused_xx_fused_ff_fused_outer * 1572864) + (((int)blockIdx.x) * 6144)) + ((((int)threadIdx.x) >> 5) * 192)) + rc)] * ph[((rc * 32) + (((int)threadIdx.x) & 31))]));
    }
  }
}

