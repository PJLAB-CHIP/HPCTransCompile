void default_function_kernel(float* T_concat, float* ph, float* ph_1, float* ph_2, float* ph_3, float* ph_4, float* ph_5, float* ph_6, float* ph_7, float* ph_8, float* ph_9, float* ph_10, float* ph_11, float* ph_12, float* ph_13, float* ph_14, float* ph_15, float* ph_16, float* ph_17, float* ph_18, float* ph_19, float* ph_20, float* ph_21, float* ph_22, float* ph_23, float* ph_24, float* ph_25) {
  float pad_temp[49152];
  float conv2d_nchw[29768];
  float conv2d_nchw_1[14400];
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 3; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 128; ++i2) {
      for (int32_t i3_outer = 0; i3_outer < 8; ++i3_outer) {
        for (int32_t i3_inner = 0; i3_inner < 16; ++i3_inner) {
          pad_temp[((((i0_i1_fused * 16384) + (i2 * 128)) + (i3_outer * 16)) + i3_inner)] = ph_13[((((i0_i1_fused * 16384) + (i2 * 128)) + (i3_outer * 16)) + i3_inner)];
        }
      }
    }
  }
  for (int32_t yy = 0; yy < 122; ++yy) {
    for (int32_t xx_outer = 0; xx_outer < 8; ++xx_outer) {
      for (int32_t xx_inner = 0; xx_inner < 16; ++xx_inner) {
        if (((xx_outer * 8) + (xx_inner >> 1)) < 61) {
          conv2d_nchw[(((yy * 122) + (xx_outer * 16)) + xx_inner)] = 0.000000e+00f;
        }
        if (((xx_outer * 8) + (xx_inner >> 1)) < 61) {
          for (int32_t rc = 0; rc < 3; ++rc) {
            for (int32_t ry = 0; ry < 7; ++ry) {
              for (int32_t rx = 0; rx < 7; ++rx) {
                conv2d_nchw[(((yy * 122) + (xx_outer * 16)) + xx_inner)] = (conv2d_nchw[(((yy * 122) + (xx_outer * 16)) + xx_inner)] + (pad_temp[((((((rc * 16384) + (yy * 128)) + (ry * 128)) + (xx_outer * 16)) + xx_inner) + rx)] * ph_25[(((rc * 49) + (ry * 7)) + rx)]));
              }
            }
          }
        }
      }
    }
  }
  for (int32_t i2_1 = 0; i2_1 < 122; ++i2_1) {
    for (int32_t i3_outer_1 = 0; i3_outer_1 < 8; ++i3_outer_1) {
      for (int32_t i3_inner_1 = 0; i3_inner_1 < 16; ++i3_inner_1) {
        if (((i3_outer_1 * 8) + (i3_inner_1 >> 1)) < 61) {
          conv2d_nchw[(((i2_1 * 122) + (i3_outer_1 * 16)) + i3_inner_1)] = max(conv2d_nchw[(((i2_1 * 122) + (i3_outer_1 * 16)) + i3_inner_1)], 0.000000e+00f);
        }
      }
    }
  }
  for (int32_t ax2 = 0; ax2 < 124; ++ax2) {
    for (int32_t ax3_outer = 0; ax3_outer < 8; ++ax3_outer) {
      for (int32_t ax3_inner = 0; ax3_inner < 16; ++ax3_inner) {
        if (((ax3_outer * 4) + (ax3_inner >> 2)) < 31) {
          pad_temp[(((ax2 * 124) + (ax3_outer * 16)) + ax3_inner)] = (((((1 <= ax2) && (ax2 < 123)) && (1 <= ((ax3_outer * 16) + ax3_inner))) && (((ax3_outer * 16) + ax3_inner) < 123)) ? conv2d_nchw[((((ax2 * 122) + (ax3_outer * 16)) + ax3_inner) - 123)] : -3.402823e+38f);
        }
      }
    }
  }
  for (int32_t ax2_1 = 0; ax2_1 < 122; ++ax2_1) {
    for (int32_t ax3_outer_1 = 0; ax3_outer_1 < 8; ++ax3_outer_1) {
      for (int32_t ax3_inner_1 = 0; ax3_inner_1 < 16; ++ax3_inner_1) {
        if (((ax3_outer_1 * 8) + (ax3_inner_1 >> 1)) < 61) {
          conv2d_nchw[(((ax2_1 * 122) + (ax3_outer_1 * 16)) + ax3_inner_1)] = -3.402823e+38f;
        }
        if (((ax3_outer_1 * 8) + (ax3_inner_1 >> 1)) < 61) {
          for (int32_t rv0 = 0; rv0 < 3; ++rv0) {
            for (int32_t rv1 = 0; rv1 < 3; ++rv1) {
              conv2d_nchw[(((ax2_1 * 122) + (ax3_outer_1 * 16)) + ax3_inner_1)] = max(conv2d_nchw[(((ax2_1 * 122) + (ax3_outer_1 * 16)) + ax3_inner_1)], pad_temp[(((((ax2_1 * 124) + (rv0 * 124)) + (ax3_outer_1 * 16)) + ax3_inner_1) + rv1)]);
            }
          }
        }
      }
    }
  }
  for (int32_t yy_1 = 0; yy_1 < 122; ++yy_1) {
    for (int32_t xx_outer_1 = 0; xx_outer_1 < 8; ++xx_outer_1) {
      for (int32_t xx_inner_1 = 0; xx_inner_1 < 16; ++xx_inner_1) {
        if (((xx_outer_1 * 8) + (xx_inner_1 >> 1)) < 61) {
          pad_temp[(((yy_1 * 122) + (xx_outer_1 * 16)) + xx_inner_1)] = 0.000000e+00f;
        }
        if (((xx_outer_1 * 8) + (xx_inner_1 >> 1)) < 61) {
          pad_temp[(((yy_1 * 122) + (xx_outer_1 * 16)) + xx_inner_1)] = (pad_temp[(((yy_1 * 122) + (xx_outer_1 * 16)) + xx_inner_1)] + (conv2d_nchw[(((yy_1 * 122) + (xx_outer_1 * 16)) + xx_inner_1)] * ph_24[0]));
        }
      }
    }
  }
  for (int32_t i2_2 = 0; i2_2 < 122; ++i2_2) {
    for (int32_t i3_outer_2 = 0; i3_outer_2 < 8; ++i3_outer_2) {
      for (int32_t i3_inner_2 = 0; i3_inner_2 < 16; ++i3_inner_2) {
        if (((i3_outer_2 * 8) + (i3_inner_2 >> 1)) < 61) {
          conv2d_nchw[(((i2_2 * 122) + (i3_outer_2 * 16)) + i3_inner_2)] = max(pad_temp[(((i2_2 * 122) + (i3_outer_2 * 16)) + i3_inner_2)], 0.000000e+00f);
        }
      }
    }
  }
  for (int32_t yy_2 = 0; yy_2 < 122; ++yy_2) {
    for (int32_t xx_outer_2 = 0; xx_outer_2 < 8; ++xx_outer_2) {
      for (int32_t xx_inner_2 = 0; xx_inner_2 < 16; ++xx_inner_2) {
        if (((xx_outer_2 * 8) + (xx_inner_2 >> 1)) < 61) {
          pad_temp[(((yy_2 * 122) + (xx_outer_2 * 16)) + xx_inner_2)] = 0.000000e+00f;
        }
        if (((xx_outer_2 * 8) + (xx_inner_2 >> 1)) < 61) {
          pad_temp[(((yy_2 * 122) + (xx_outer_2 * 16)) + xx_inner_2)] = (pad_temp[(((yy_2 * 122) + (xx_outer_2 * 16)) + xx_inner_2)] + (conv2d_nchw[(((yy_2 * 122) + (xx_outer_2 * 16)) + xx_inner_2)] * ph_23[0]));
        }
      }
    }
  }
  for (int32_t i2_3 = 0; i2_3 < 122; ++i2_3) {
    for (int32_t i3_outer_3 = 0; i3_outer_3 < 8; ++i3_outer_3) {
      for (int32_t i3_inner_3 = 0; i3_inner_3 < 16; ++i3_inner_3) {
        if (((i3_outer_3 * 8) + (i3_inner_3 >> 1)) < 61) {
          conv2d_nchw[(((i2_3 * 122) + (i3_outer_3 * 16)) + i3_inner_3)] = max(pad_temp[(((i2_3 * 122) + (i3_outer_3 * 16)) + i3_inner_3)], 0.000000e+00f);
        }
      }
    }
  }
  for (int32_t i2_4 = 0; i2_4 < 122; ++i2_4) {
    for (int32_t i3_outer_4 = 0; i3_outer_4 < 8; ++i3_outer_4) {
      for (int32_t i3_inner_4 = 0; i3_inner_4 < 16; ++i3_inner_4) {
        if (((i3_outer_4 * 8) + (i3_inner_4 >> 1)) < 61) {
          pad_temp[(((i2_4 * 122) + (i3_outer_4 * 16)) + i3_inner_4)] = conv2d_nchw[(((i2_4 * 122) + (i3_outer_4 * 16)) + i3_inner_4)];
        }
      }
    }
  }
  for (int32_t yy_3 = 0; yy_3 < 120; ++yy_3) {
    for (int32_t xx_outer_3 = 0; xx_outer_3 < 8; ++xx_outer_3) {
      for (int32_t xx_inner_3 = 0; xx_inner_3 < 16; ++xx_inner_3) {
        if (((xx_outer_3 * 2) + (xx_inner_3 >> 3)) < 15) {
          conv2d_nchw_1[(((yy_3 * 120) + (xx_outer_3 * 16)) + xx_inner_3)] = 0.000000e+00f;
        }
        if (((xx_outer_3 * 2) + (xx_inner_3 >> 3)) < 15) {
          for (int32_t ry_1 = 0; ry_1 < 3; ++ry_1) {
            for (int32_t rx_1 = 0; rx_1 < 3; ++rx_1) {
              conv2d_nchw_1[(((yy_3 * 120) + (xx_outer_3 * 16)) + xx_inner_3)] = (conv2d_nchw_1[(((yy_3 * 120) + (xx_outer_3 * 16)) + xx_inner_3)] + (pad_temp[(((((yy_3 * 122) + (ry_1 * 122)) + (xx_outer_3 * 16)) + xx_inner_3) + rx_1)] * ph_22[((ry_1 * 3) + rx_1)]));
            }
          }
        }
      }
    }
  }
  for (int32_t i2_5 = 0; i2_5 < 120; ++i2_5) {
    for (int32_t i3_outer_5 = 0; i3_outer_5 < 8; ++i3_outer_5) {
      for (int32_t i3_inner_5 = 0; i3_inner_5 < 16; ++i3_inner_5) {
        if (((i3_outer_5 * 2) + (i3_inner_5 >> 3)) < 15) {
          conv2d_nchw_1[(((i2_5 * 120) + (i3_outer_5 * 16)) + i3_inner_5)] = max(conv2d_nchw_1[(((i2_5 * 120) + (i3_outer_5 * 16)) + i3_inner_5)], 0.000000e+00f);
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 2; ++ax0_ax1_fused) {
    for (int32_t ax2_2 = 0; ax2_2 < 122; ++ax2_2) {
      for (int32_t ax3_outer_2 = 0; ax3_outer_2 < 8; ++ax3_outer_2) {
        for (int32_t ax3_inner_2 = 0; ax3_inner_2 < 16; ++ax3_inner_2) {
          if (((ax3_outer_2 * 8) + (ax3_inner_2 >> 1)) < 61) {
            pad_temp[((((ax0_ax1_fused * 14884) + (ax2_2 * 122)) + (ax3_outer_2 * 16)) + ax3_inner_2)] = ((ax0_ax1_fused == 1) ? conv2d_nchw_1[(((((((ax0_ax1_fused + 1) >> 1) * 14400) + (ax2_2 * 120)) + (ax3_outer_2 * 16)) + ax3_inner_2) - 14400)] : conv2d_nchw[(((ax2_2 * 122) + (ax3_outer_2 * 16)) + ax3_inner_2)]);
          }
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_1 = 0; i0_i1_fused_1 < 2; ++i0_i1_fused_1) {
    for (int32_t i2_6 = 0; i2_6 < 122; ++i2_6) {
      for (int32_t i3_outer_6 = 0; i3_outer_6 < 8; ++i3_outer_6) {
        for (int32_t i3_inner_6 = 0; i3_inner_6 < 16; ++i3_inner_6) {
          if (((i3_outer_6 * 8) + (i3_inner_6 >> 1)) < 61) {
            conv2d_nchw[((((i0_i1_fused_1 * 14884) + (i2_6 * 122)) + (i3_outer_6 * 16)) + i3_inner_6)] = pad_temp[((((i0_i1_fused_1 * 14884) + (i2_6 * 122)) + (i3_outer_6 * 16)) + i3_inner_6)];
          }
        }
      }
    }
  }
  for (int32_t yy_4 = 0; yy_4 < 122; ++yy_4) {
    for (int32_t xx_outer_4 = 0; xx_outer_4 < 8; ++xx_outer_4) {
      for (int32_t xx_inner_4 = 0; xx_inner_4 < 16; ++xx_inner_4) {
        if (((xx_outer_4 * 8) + (xx_inner_4 >> 1)) < 61) {
          pad_temp[(((yy_4 * 122) + (xx_outer_4 * 16)) + xx_inner_4)] = 0.000000e+00f;
        }
        if (((xx_outer_4 * 8) + (xx_inner_4 >> 1)) < 61) {
          for (int32_t rc_1 = 0; rc_1 < 2; ++rc_1) {
            pad_temp[(((yy_4 * 122) + (xx_outer_4 * 16)) + xx_inner_4)] = (pad_temp[(((yy_4 * 122) + (xx_outer_4 * 16)) + xx_inner_4)] + (conv2d_nchw[((((rc_1 * 14884) + (yy_4 * 122)) + (xx_outer_4 * 16)) + xx_inner_4)] * ph_21[rc_1]));
          }
        }
      }
    }
  }
  for (int32_t i2_7 = 0; i2_7 < 122; ++i2_7) {
    for (int32_t i3_outer_7 = 0; i3_outer_7 < 8; ++i3_outer_7) {
      for (int32_t i3_inner_7 = 0; i3_inner_7 < 16; ++i3_inner_7) {
        if (((i3_outer_7 * 8) + (i3_inner_7 >> 1)) < 61) {
          conv2d_nchw[(((i2_7 * 122) + (i3_outer_7 * 16)) + i3_inner_7)] = max(pad_temp[(((i2_7 * 122) + (i3_outer_7 * 16)) + i3_inner_7)], 0.000000e+00f);
        }
      }
    }
  }
  for (int32_t i2_8 = 0; i2_8 < 122; ++i2_8) {
    for (int32_t i3_outer_8 = 0; i3_outer_8 < 8; ++i3_outer_8) {
      for (int32_t i3_inner_8 = 0; i3_inner_8 < 16; ++i3_inner_8) {
        if (((i3_outer_8 * 8) + (i3_inner_8 >> 1)) < 61) {
          pad_temp[(((i2_8 * 122) + (i3_outer_8 * 16)) + i3_inner_8)] = conv2d_nchw[(((i2_8 * 122) + (i3_outer_8 * 16)) + i3_inner_8)];
        }
      }
    }
  }
  for (int32_t yy_5 = 0; yy_5 < 122; ++yy_5) {
    for (int32_t xx_outer_5 = 0; xx_outer_5 < 8; ++xx_outer_5) {
      for (int32_t xx_inner_5 = 0; xx_inner_5 < 16; ++xx_inner_5) {
        if (((xx_outer_5 * 8) + (xx_inner_5 >> 1)) < 61) {
          conv2d_nchw[(((yy_5 * 122) + (xx_outer_5 * 16)) + xx_inner_5)] = 0.000000e+00f;
        }
        if (((xx_outer_5 * 8) + (xx_inner_5 >> 1)) < 61) {
          conv2d_nchw[(((yy_5 * 122) + (xx_outer_5 * 16)) + xx_inner_5)] = (conv2d_nchw[(((yy_5 * 122) + (xx_outer_5 * 16)) + xx_inner_5)] + (pad_temp[(((yy_5 * 122) + (xx_outer_5 * 16)) + xx_inner_5)] * ph_20[0]));
        }
      }
    }
  }
  for (int32_t i2_9 = 0; i2_9 < 122; ++i2_9) {
    for (int32_t i3_outer_9 = 0; i3_outer_9 < 8; ++i3_outer_9) {
      for (int32_t i3_inner_9 = 0; i3_inner_9 < 16; ++i3_inner_9) {
        if (((i3_outer_9 * 8) + (i3_inner_9 >> 1)) < 61) {
          pad_temp[(((i2_9 * 122) + (i3_outer_9 * 16)) + i3_inner_9)] = max(conv2d_nchw[(((i2_9 * 122) + (i3_outer_9 * 16)) + i3_inner_9)], 0.000000e+00f);
        }
      }
    }
  }
  for (int32_t i2_10 = 0; i2_10 < 122; ++i2_10) {
    for (int32_t i3_outer_10 = 0; i3_outer_10 < 8; ++i3_outer_10) {
      for (int32_t i3_inner_10 = 0; i3_inner_10 < 16; ++i3_inner_10) {
        if (((i3_outer_10 * 8) + (i3_inner_10 >> 1)) < 61) {
          conv2d_nchw[(((i2_10 * 122) + (i3_outer_10 * 16)) + i3_inner_10)] = pad_temp[(((i2_10 * 122) + (i3_outer_10 * 16)) + i3_inner_10)];
        }
      }
    }
  }
  for (int32_t yy_6 = 0; yy_6 < 120; ++yy_6) {
    for (int32_t xx_outer_6 = 0; xx_outer_6 < 8; ++xx_outer_6) {
      for (int32_t xx_inner_6 = 0; xx_inner_6 < 16; ++xx_inner_6) {
        if (((xx_outer_6 * 2) + (xx_inner_6 >> 3)) < 15) {
          conv2d_nchw_1[(((yy_6 * 120) + (xx_outer_6 * 16)) + xx_inner_6)] = 0.000000e+00f;
        }
        if (((xx_outer_6 * 2) + (xx_inner_6 >> 3)) < 15) {
          for (int32_t ry_2 = 0; ry_2 < 3; ++ry_2) {
            for (int32_t rx_2 = 0; rx_2 < 3; ++rx_2) {
              conv2d_nchw_1[(((yy_6 * 120) + (xx_outer_6 * 16)) + xx_inner_6)] = (conv2d_nchw_1[(((yy_6 * 120) + (xx_outer_6 * 16)) + xx_inner_6)] + (conv2d_nchw[(((((yy_6 * 122) + (ry_2 * 122)) + (xx_outer_6 * 16)) + xx_inner_6) + rx_2)] * ph_19[((ry_2 * 3) + rx_2)]));
            }
          }
        }
      }
    }
  }
  for (int32_t i2_11 = 0; i2_11 < 120; ++i2_11) {
    for (int32_t i3_outer_11 = 0; i3_outer_11 < 8; ++i3_outer_11) {
      for (int32_t i3_inner_11 = 0; i3_inner_11 < 16; ++i3_inner_11) {
        if (((i3_outer_11 * 2) + (i3_inner_11 >> 3)) < 15) {
          conv2d_nchw_1[(((i2_11 * 120) + (i3_outer_11 * 16)) + i3_inner_11)] = max(conv2d_nchw_1[(((i2_11 * 120) + (i3_outer_11 * 16)) + i3_inner_11)], 0.000000e+00f);
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < 2; ++ax0_ax1_fused_1) {
    for (int32_t ax2_3 = 0; ax2_3 < 122; ++ax2_3) {
      for (int32_t ax3_outer_3 = 0; ax3_outer_3 < 8; ++ax3_outer_3) {
        for (int32_t ax3_inner_3 = 0; ax3_inner_3 < 16; ++ax3_inner_3) {
          if (((ax3_outer_3 * 8) + (ax3_inner_3 >> 1)) < 61) {
            conv2d_nchw[((((ax0_ax1_fused_1 * 14884) + (ax2_3 * 122)) + (ax3_outer_3 * 16)) + ax3_inner_3)] = ((ax0_ax1_fused_1 == 1) ? conv2d_nchw_1[(((((((ax0_ax1_fused_1 + 1) >> 1) * 14400) + (ax2_3 * 120)) + (ax3_outer_3 * 16)) + ax3_inner_3) - 14400)] : pad_temp[(((ax2_3 * 122) + (ax3_outer_3 * 16)) + ax3_inner_3)]);
          }
        }
      }
    }
  }
  for (int32_t yy_7 = 0; yy_7 < 122; ++yy_7) {
    for (int32_t xx_outer_7 = 0; xx_outer_7 < 8; ++xx_outer_7) {
      for (int32_t xx_inner_7 = 0; xx_inner_7 < 16; ++xx_inner_7) {
        if (((xx_outer_7 * 8) + (xx_inner_7 >> 1)) < 61) {
          pad_temp[(((yy_7 * 122) + (xx_outer_7 * 16)) + xx_inner_7)] = 0.000000e+00f;
        }
        if (((xx_outer_7 * 8) + (xx_inner_7 >> 1)) < 61) {
          for (int32_t rc_2 = 0; rc_2 < 2; ++rc_2) {
            pad_temp[(((yy_7 * 122) + (xx_outer_7 * 16)) + xx_inner_7)] = (pad_temp[(((yy_7 * 122) + (xx_outer_7 * 16)) + xx_inner_7)] + (conv2d_nchw[((((rc_2 * 14884) + (yy_7 * 122)) + (xx_outer_7 * 16)) + xx_inner_7)] * ph_18[rc_2]));
          }
        }
      }
    }
  }
  for (int32_t i2_12 = 0; i2_12 < 122; ++i2_12) {
    for (int32_t i3_outer_12 = 0; i3_outer_12 < 8; ++i3_outer_12) {
      for (int32_t i3_inner_12 = 0; i3_inner_12 < 16; ++i3_inner_12) {
        if (((i3_outer_12 * 8) + (i3_inner_12 >> 1)) < 61) {
          conv2d_nchw[(((i2_12 * 122) + (i3_outer_12 * 16)) + i3_inner_12)] = max(pad_temp[(((i2_12 * 122) + (i3_outer_12 * 16)) + i3_inner_12)], 0.000000e+00f);
        }
      }
    }
  }
  for (int32_t i2_13 = 0; i2_13 < 122; ++i2_13) {
    for (int32_t i3_outer_13 = 0; i3_outer_13 < 8; ++i3_outer_13) {
      for (int32_t i3_inner_13 = 0; i3_inner_13 < 16; ++i3_inner_13) {
        if (((i3_outer_13 * 8) + (i3_inner_13 >> 1)) < 61) {
          pad_temp[(((i2_13 * 122) + (i3_outer_13 * 16)) + i3_inner_13)] = conv2d_nchw[(((i2_13 * 122) + (i3_outer_13 * 16)) + i3_inner_13)];
        }
      }
    }
  }
  for (int32_t yy_8 = 0; yy_8 < 122; ++yy_8) {
    for (int32_t xx_outer_8 = 0; xx_outer_8 < 8; ++xx_outer_8) {
      for (int32_t xx_inner_8 = 0; xx_inner_8 < 16; ++xx_inner_8) {
        if (((xx_outer_8 * 8) + (xx_inner_8 >> 1)) < 61) {
          conv2d_nchw[(((yy_8 * 122) + (xx_outer_8 * 16)) + xx_inner_8)] = 0.000000e+00f;
        }
        if (((xx_outer_8 * 8) + (xx_inner_8 >> 1)) < 61) {
          conv2d_nchw[(((yy_8 * 122) + (xx_outer_8 * 16)) + xx_inner_8)] = (conv2d_nchw[(((yy_8 * 122) + (xx_outer_8 * 16)) + xx_inner_8)] + (pad_temp[(((yy_8 * 122) + (xx_outer_8 * 16)) + xx_inner_8)] * ph_17[0]));
        }
      }
    }
  }
  for (int32_t i2_14 = 0; i2_14 < 122; ++i2_14) {
    for (int32_t i3_outer_14 = 0; i3_outer_14 < 8; ++i3_outer_14) {
      for (int32_t i3_inner_14 = 0; i3_inner_14 < 16; ++i3_inner_14) {
        if (((i3_outer_14 * 8) + (i3_inner_14 >> 1)) < 61) {
          pad_temp[(((i2_14 * 122) + (i3_outer_14 * 16)) + i3_inner_14)] = max(conv2d_nchw[(((i2_14 * 122) + (i3_outer_14 * 16)) + i3_inner_14)], 0.000000e+00f);
        }
      }
    }
  }
  for (int32_t i2_15 = 0; i2_15 < 122; ++i2_15) {
    for (int32_t i3_outer_15 = 0; i3_outer_15 < 8; ++i3_outer_15) {
      for (int32_t i3_inner_15 = 0; i3_inner_15 < 16; ++i3_inner_15) {
        if (((i3_outer_15 * 8) + (i3_inner_15 >> 1)) < 61) {
          conv2d_nchw[(((i2_15 * 122) + (i3_outer_15 * 16)) + i3_inner_15)] = pad_temp[(((i2_15 * 122) + (i3_outer_15 * 16)) + i3_inner_15)];
        }
      }
    }
  }
  for (int32_t yy_9 = 0; yy_9 < 120; ++yy_9) {
    for (int32_t xx_outer_9 = 0; xx_outer_9 < 8; ++xx_outer_9) {
      for (int32_t xx_inner_9 = 0; xx_inner_9 < 16; ++xx_inner_9) {
        if (((xx_outer_9 * 2) + (xx_inner_9 >> 3)) < 15) {
          conv2d_nchw_1[(((yy_9 * 120) + (xx_outer_9 * 16)) + xx_inner_9)] = 0.000000e+00f;
        }
        if (((xx_outer_9 * 2) + (xx_inner_9 >> 3)) < 15) {
          for (int32_t ry_3 = 0; ry_3 < 3; ++ry_3) {
            for (int32_t rx_3 = 0; rx_3 < 3; ++rx_3) {
              conv2d_nchw_1[(((yy_9 * 120) + (xx_outer_9 * 16)) + xx_inner_9)] = (conv2d_nchw_1[(((yy_9 * 120) + (xx_outer_9 * 16)) + xx_inner_9)] + (conv2d_nchw[(((((yy_9 * 122) + (ry_3 * 122)) + (xx_outer_9 * 16)) + xx_inner_9) + rx_3)] * ph_16[((ry_3 * 3) + rx_3)]));
            }
          }
        }
      }
    }
  }
  for (int32_t i2_16 = 0; i2_16 < 120; ++i2_16) {
    for (int32_t i3_outer_16 = 0; i3_outer_16 < 8; ++i3_outer_16) {
      for (int32_t i3_inner_16 = 0; i3_inner_16 < 16; ++i3_inner_16) {
        if (((i3_outer_16 * 2) + (i3_inner_16 >> 3)) < 15) {
          conv2d_nchw_1[(((i2_16 * 120) + (i3_outer_16 * 16)) + i3_inner_16)] = max(conv2d_nchw_1[(((i2_16 * 120) + (i3_outer_16 * 16)) + i3_inner_16)], 0.000000e+00f);
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_2 = 0; ax0_ax1_fused_2 < 2; ++ax0_ax1_fused_2) {
    for (int32_t ax2_4 = 0; ax2_4 < 122; ++ax2_4) {
      for (int32_t ax3_outer_4 = 0; ax3_outer_4 < 8; ++ax3_outer_4) {
        for (int32_t ax3_inner_4 = 0; ax3_inner_4 < 16; ++ax3_inner_4) {
          if (((ax3_outer_4 * 8) + (ax3_inner_4 >> 1)) < 61) {
            conv2d_nchw[((((ax0_ax1_fused_2 * 14884) + (ax2_4 * 122)) + (ax3_outer_4 * 16)) + ax3_inner_4)] = ((ax0_ax1_fused_2 == 1) ? conv2d_nchw_1[(((((((ax0_ax1_fused_2 + 1) >> 1) * 14400) + (ax2_4 * 120)) + (ax3_outer_4 * 16)) + ax3_inner_4) - 14400)] : pad_temp[(((ax2_4 * 122) + (ax3_outer_4 * 16)) + ax3_inner_4)]);
          }
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_3 = 0; ax0_ax1_fused_3 < 2; ++ax0_ax1_fused_3) {
    for (int32_t ax2_5 = 0; ax2_5 < 124; ++ax2_5) {
      for (int32_t ax3_outer_5 = 0; ax3_outer_5 < 8; ++ax3_outer_5) {
        for (int32_t ax3_inner_5 = 0; ax3_inner_5 < 16; ++ax3_inner_5) {
          if (((ax3_outer_5 * 4) + (ax3_inner_5 >> 2)) < 31) {
            pad_temp[((((ax0_ax1_fused_3 * 15376) + (ax2_5 * 124)) + (ax3_outer_5 * 16)) + ax3_inner_5)] = (((((1 <= ax2_5) && (ax2_5 < 123)) && (1 <= ((ax3_outer_5 * 16) + ax3_inner_5))) && (((ax3_outer_5 * 16) + ax3_inner_5) < 123)) ? conv2d_nchw[(((((ax0_ax1_fused_3 * 14884) + (ax2_5 * 122)) + (ax3_outer_5 * 16)) + ax3_inner_5) - 123)] : -3.402823e+38f);
          }
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_4 = 0; ax0_ax1_fused_4 < 2; ++ax0_ax1_fused_4) {
    for (int32_t ax2_6 = 0; ax2_6 < 122; ++ax2_6) {
      for (int32_t ax3_outer_6 = 0; ax3_outer_6 < 8; ++ax3_outer_6) {
        for (int32_t ax3_inner_6 = 0; ax3_inner_6 < 16; ++ax3_inner_6) {
          if (((ax3_outer_6 * 8) + (ax3_inner_6 >> 1)) < 61) {
            conv2d_nchw[((((ax0_ax1_fused_4 * 14884) + (ax2_6 * 122)) + (ax3_outer_6 * 16)) + ax3_inner_6)] = -3.402823e+38f;
          }
          if (((ax3_outer_6 * 8) + (ax3_inner_6 >> 1)) < 61) {
            for (int32_t rv0_1 = 0; rv0_1 < 3; ++rv0_1) {
              for (int32_t rv1_1 = 0; rv1_1 < 3; ++rv1_1) {
                conv2d_nchw[((((ax0_ax1_fused_4 * 14884) + (ax2_6 * 122)) + (ax3_outer_6 * 16)) + ax3_inner_6)] = max(conv2d_nchw[((((ax0_ax1_fused_4 * 14884) + (ax2_6 * 122)) + (ax3_outer_6 * 16)) + ax3_inner_6)], pad_temp[((((((ax0_ax1_fused_4 * 15376) + (ax2_6 * 124)) + (rv0_1 * 124)) + (ax3_outer_6 * 16)) + ax3_inner_6) + rv1_1)]);
              }
            }
          }
        }
      }
    }
  }
  for (int32_t yy_10 = 0; yy_10 < 122; ++yy_10) {
    for (int32_t xx_outer_10 = 0; xx_outer_10 < 8; ++xx_outer_10) {
      for (int32_t xx_inner_10 = 0; xx_inner_10 < 16; ++xx_inner_10) {
        if (((xx_outer_10 * 8) + (xx_inner_10 >> 1)) < 61) {
          pad_temp[(((yy_10 * 122) + (xx_outer_10 * 16)) + xx_inner_10)] = 0.000000e+00f;
        }
        if (((xx_outer_10 * 8) + (xx_inner_10 >> 1)) < 61) {
          for (int32_t rc_3 = 0; rc_3 < 2; ++rc_3) {
            pad_temp[(((yy_10 * 122) + (xx_outer_10 * 16)) + xx_inner_10)] = (pad_temp[(((yy_10 * 122) + (xx_outer_10 * 16)) + xx_inner_10)] + (conv2d_nchw[((((rc_3 * 14884) + (yy_10 * 122)) + (xx_outer_10 * 16)) + xx_inner_10)] * ph_15[rc_3]));
          }
        }
      }
    }
  }
  for (int32_t i2_17 = 0; i2_17 < 122; ++i2_17) {
    for (int32_t i3_outer_17 = 0; i3_outer_17 < 8; ++i3_outer_17) {
      for (int32_t i3_inner_17 = 0; i3_inner_17 < 16; ++i3_inner_17) {
        if (((i3_outer_17 * 8) + (i3_inner_17 >> 1)) < 61) {
          conv2d_nchw[(((i2_17 * 122) + (i3_outer_17 * 16)) + i3_inner_17)] = max(pad_temp[(((i2_17 * 122) + (i3_outer_17 * 16)) + i3_inner_17)], 0.000000e+00f);
        }
      }
    }
  }
  for (int32_t i2_18 = 0; i2_18 < 122; ++i2_18) {
    for (int32_t i3_outer_18 = 0; i3_outer_18 < 8; ++i3_outer_18) {
      for (int32_t i3_inner_18 = 0; i3_inner_18 < 16; ++i3_inner_18) {
        if (((i3_outer_18 * 8) + (i3_inner_18 >> 1)) < 61) {
          pad_temp[(((i2_18 * 122) + (i3_outer_18 * 16)) + i3_inner_18)] = conv2d_nchw[(((i2_18 * 122) + (i3_outer_18 * 16)) + i3_inner_18)];
        }
      }
    }
  }
  for (int32_t yy_11 = 0; yy_11 < 122; ++yy_11) {
    for (int32_t xx_outer_11 = 0; xx_outer_11 < 8; ++xx_outer_11) {
      for (int32_t xx_inner_11 = 0; xx_inner_11 < 16; ++xx_inner_11) {
        if (((xx_outer_11 * 8) + (xx_inner_11 >> 1)) < 61) {
          conv2d_nchw[(((yy_11 * 122) + (xx_outer_11 * 16)) + xx_inner_11)] = 0.000000e+00f;
        }
        if (((xx_outer_11 * 8) + (xx_inner_11 >> 1)) < 61) {
          conv2d_nchw[(((yy_11 * 122) + (xx_outer_11 * 16)) + xx_inner_11)] = (conv2d_nchw[(((yy_11 * 122) + (xx_outer_11 * 16)) + xx_inner_11)] + (pad_temp[(((yy_11 * 122) + (xx_outer_11 * 16)) + xx_inner_11)] * ph_14[0]));
        }
      }
    }
  }
  for (int32_t i2_19 = 0; i2_19 < 122; ++i2_19) {
    for (int32_t i3_outer_19 = 0; i3_outer_19 < 8; ++i3_outer_19) {
      for (int32_t i3_inner_19 = 0; i3_inner_19 < 16; ++i3_inner_19) {
        if (((i3_outer_19 * 8) + (i3_inner_19 >> 1)) < 61) {
          pad_temp[(((i2_19 * 122) + (i3_outer_19 * 16)) + i3_inner_19)] = max(conv2d_nchw[(((i2_19 * 122) + (i3_outer_19 * 16)) + i3_inner_19)], 0.000000e+00f);
        }
      }
    }
  }
  for (int32_t i2_20 = 0; i2_20 < 122; ++i2_20) {
    for (int32_t i3_outer_20 = 0; i3_outer_20 < 8; ++i3_outer_20) {
      for (int32_t i3_inner_20 = 0; i3_inner_20 < 16; ++i3_inner_20) {
        if (((i3_outer_20 * 8) + (i3_inner_20 >> 1)) < 61) {
          conv2d_nchw[(((i2_20 * 122) + (i3_outer_20 * 16)) + i3_inner_20)] = pad_temp[(((i2_20 * 122) + (i3_outer_20 * 16)) + i3_inner_20)];
        }
      }
    }
  }
  for (int32_t yy_12 = 0; yy_12 < 120; ++yy_12) {
    for (int32_t xx_outer_12 = 0; xx_outer_12 < 8; ++xx_outer_12) {
      for (int32_t xx_inner_12 = 0; xx_inner_12 < 16; ++xx_inner_12) {
        if (((xx_outer_12 * 2) + (xx_inner_12 >> 3)) < 15) {
          conv2d_nchw_1[(((yy_12 * 120) + (xx_outer_12 * 16)) + xx_inner_12)] = 0.000000e+00f;
        }
        if (((xx_outer_12 * 2) + (xx_inner_12 >> 3)) < 15) {
          for (int32_t ry_4 = 0; ry_4 < 3; ++ry_4) {
            for (int32_t rx_4 = 0; rx_4 < 3; ++rx_4) {
              conv2d_nchw_1[(((yy_12 * 120) + (xx_outer_12 * 16)) + xx_inner_12)] = (conv2d_nchw_1[(((yy_12 * 120) + (xx_outer_12 * 16)) + xx_inner_12)] + (conv2d_nchw[(((((yy_12 * 122) + (ry_4 * 122)) + (xx_outer_12 * 16)) + xx_inner_12) + rx_4)] * ph[((ry_4 * 3) + rx_4)]));
            }
          }
        }
      }
    }
  }
  for (int32_t i2_21 = 0; i2_21 < 120; ++i2_21) {
    for (int32_t i3_outer_21 = 0; i3_outer_21 < 8; ++i3_outer_21) {
      for (int32_t i3_inner_21 = 0; i3_inner_21 < 16; ++i3_inner_21) {
        if (((i3_outer_21 * 2) + (i3_inner_21 >> 3)) < 15) {
          conv2d_nchw_1[(((i2_21 * 120) + (i3_outer_21 * 16)) + i3_inner_21)] = max(conv2d_nchw_1[(((i2_21 * 120) + (i3_outer_21 * 16)) + i3_inner_21)], 0.000000e+00f);
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_5 = 0; ax0_ax1_fused_5 < 2; ++ax0_ax1_fused_5) {
    for (int32_t ax2_7 = 0; ax2_7 < 122; ++ax2_7) {
      for (int32_t ax3_outer_7 = 0; ax3_outer_7 < 8; ++ax3_outer_7) {
        for (int32_t ax3_inner_7 = 0; ax3_inner_7 < 16; ++ax3_inner_7) {
          if (((ax3_outer_7 * 8) + (ax3_inner_7 >> 1)) < 61) {
            conv2d_nchw[((((ax0_ax1_fused_5 * 14884) + (ax2_7 * 122)) + (ax3_outer_7 * 16)) + ax3_inner_7)] = ((ax0_ax1_fused_5 == 1) ? conv2d_nchw_1[(((((((ax0_ax1_fused_5 + 1) >> 1) * 14400) + (ax2_7 * 120)) + (ax3_outer_7 * 16)) + ax3_inner_7) - 14400)] : pad_temp[(((ax2_7 * 122) + (ax3_outer_7 * 16)) + ax3_inner_7)]);
          }
        }
      }
    }
  }
  for (int32_t yy_13 = 0; yy_13 < 122; ++yy_13) {
    for (int32_t xx_outer_13 = 0; xx_outer_13 < 8; ++xx_outer_13) {
      for (int32_t xx_inner_13 = 0; xx_inner_13 < 16; ++xx_inner_13) {
        if (((xx_outer_13 * 8) + (xx_inner_13 >> 1)) < 61) {
          pad_temp[(((yy_13 * 122) + (xx_outer_13 * 16)) + xx_inner_13)] = 0.000000e+00f;
        }
        if (((xx_outer_13 * 8) + (xx_inner_13 >> 1)) < 61) {
          for (int32_t rc_4 = 0; rc_4 < 2; ++rc_4) {
            pad_temp[(((yy_13 * 122) + (xx_outer_13 * 16)) + xx_inner_13)] = (pad_temp[(((yy_13 * 122) + (xx_outer_13 * 16)) + xx_inner_13)] + (conv2d_nchw[((((rc_4 * 14884) + (yy_13 * 122)) + (xx_outer_13 * 16)) + xx_inner_13)] * ph_12[rc_4]));
          }
        }
      }
    }
  }
  for (int32_t i2_22 = 0; i2_22 < 122; ++i2_22) {
    for (int32_t i3_outer_22 = 0; i3_outer_22 < 8; ++i3_outer_22) {
      for (int32_t i3_inner_22 = 0; i3_inner_22 < 16; ++i3_inner_22) {
        if (((i3_outer_22 * 8) + (i3_inner_22 >> 1)) < 61) {
          conv2d_nchw[(((i2_22 * 122) + (i3_outer_22 * 16)) + i3_inner_22)] = max(pad_temp[(((i2_22 * 122) + (i3_outer_22 * 16)) + i3_inner_22)], 0.000000e+00f);
        }
      }
    }
  }
  for (int32_t i2_23 = 0; i2_23 < 122; ++i2_23) {
    for (int32_t i3_outer_23 = 0; i3_outer_23 < 8; ++i3_outer_23) {
      for (int32_t i3_inner_23 = 0; i3_inner_23 < 16; ++i3_inner_23) {
        if (((i3_outer_23 * 8) + (i3_inner_23 >> 1)) < 61) {
          pad_temp[(((i2_23 * 122) + (i3_outer_23 * 16)) + i3_inner_23)] = conv2d_nchw[(((i2_23 * 122) + (i3_outer_23 * 16)) + i3_inner_23)];
        }
      }
    }
  }
  for (int32_t yy_14 = 0; yy_14 < 122; ++yy_14) {
    for (int32_t xx_outer_14 = 0; xx_outer_14 < 8; ++xx_outer_14) {
      for (int32_t xx_inner_14 = 0; xx_inner_14 < 16; ++xx_inner_14) {
        if (((xx_outer_14 * 8) + (xx_inner_14 >> 1)) < 61) {
          conv2d_nchw[(((yy_14 * 122) + (xx_outer_14 * 16)) + xx_inner_14)] = 0.000000e+00f;
        }
        if (((xx_outer_14 * 8) + (xx_inner_14 >> 1)) < 61) {
          conv2d_nchw[(((yy_14 * 122) + (xx_outer_14 * 16)) + xx_inner_14)] = (conv2d_nchw[(((yy_14 * 122) + (xx_outer_14 * 16)) + xx_inner_14)] + (pad_temp[(((yy_14 * 122) + (xx_outer_14 * 16)) + xx_inner_14)] * ph_11[0]));
        }
      }
    }
  }
  for (int32_t i2_24 = 0; i2_24 < 122; ++i2_24) {
    for (int32_t i3_outer_24 = 0; i3_outer_24 < 8; ++i3_outer_24) {
      for (int32_t i3_inner_24 = 0; i3_inner_24 < 16; ++i3_inner_24) {
        if (((i3_outer_24 * 8) + (i3_inner_24 >> 1)) < 61) {
          pad_temp[(((i2_24 * 122) + (i3_outer_24 * 16)) + i3_inner_24)] = max(conv2d_nchw[(((i2_24 * 122) + (i3_outer_24 * 16)) + i3_inner_24)], 0.000000e+00f);
        }
      }
    }
  }
  for (int32_t i2_25 = 0; i2_25 < 122; ++i2_25) {
    for (int32_t i3_outer_25 = 0; i3_outer_25 < 8; ++i3_outer_25) {
      for (int32_t i3_inner_25 = 0; i3_inner_25 < 16; ++i3_inner_25) {
        if (((i3_outer_25 * 8) + (i3_inner_25 >> 1)) < 61) {
          conv2d_nchw[(((i2_25 * 122) + (i3_outer_25 * 16)) + i3_inner_25)] = pad_temp[(((i2_25 * 122) + (i3_outer_25 * 16)) + i3_inner_25)];
        }
      }
    }
  }
  for (int32_t yy_15 = 0; yy_15 < 120; ++yy_15) {
    for (int32_t xx_outer_15 = 0; xx_outer_15 < 8; ++xx_outer_15) {
      for (int32_t xx_inner_15 = 0; xx_inner_15 < 16; ++xx_inner_15) {
        if (((xx_outer_15 * 2) + (xx_inner_15 >> 3)) < 15) {
          conv2d_nchw_1[(((yy_15 * 120) + (xx_outer_15 * 16)) + xx_inner_15)] = 0.000000e+00f;
        }
        if (((xx_outer_15 * 2) + (xx_inner_15 >> 3)) < 15) {
          for (int32_t ry_5 = 0; ry_5 < 3; ++ry_5) {
            for (int32_t rx_5 = 0; rx_5 < 3; ++rx_5) {
              conv2d_nchw_1[(((yy_15 * 120) + (xx_outer_15 * 16)) + xx_inner_15)] = (conv2d_nchw_1[(((yy_15 * 120) + (xx_outer_15 * 16)) + xx_inner_15)] + (conv2d_nchw[(((((yy_15 * 122) + (ry_5 * 122)) + (xx_outer_15 * 16)) + xx_inner_15) + rx_5)] * ph_10[((ry_5 * 3) + rx_5)]));
            }
          }
        }
      }
    }
  }
  for (int32_t i2_26 = 0; i2_26 < 120; ++i2_26) {
    for (int32_t i3_outer_26 = 0; i3_outer_26 < 8; ++i3_outer_26) {
      for (int32_t i3_inner_26 = 0; i3_inner_26 < 16; ++i3_inner_26) {
        if (((i3_outer_26 * 2) + (i3_inner_26 >> 3)) < 15) {
          conv2d_nchw_1[(((i2_26 * 120) + (i3_outer_26 * 16)) + i3_inner_26)] = max(conv2d_nchw_1[(((i2_26 * 120) + (i3_outer_26 * 16)) + i3_inner_26)], 0.000000e+00f);
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_6 = 0; ax0_ax1_fused_6 < 2; ++ax0_ax1_fused_6) {
    for (int32_t ax2_8 = 0; ax2_8 < 122; ++ax2_8) {
      for (int32_t ax3_outer_8 = 0; ax3_outer_8 < 8; ++ax3_outer_8) {
        for (int32_t ax3_inner_8 = 0; ax3_inner_8 < 16; ++ax3_inner_8) {
          if (((ax3_outer_8 * 8) + (ax3_inner_8 >> 1)) < 61) {
            conv2d_nchw[((((ax0_ax1_fused_6 * 14884) + (ax2_8 * 122)) + (ax3_outer_8 * 16)) + ax3_inner_8)] = ((ax0_ax1_fused_6 == 1) ? conv2d_nchw_1[(((((((ax0_ax1_fused_6 + 1) >> 1) * 14400) + (ax2_8 * 120)) + (ax3_outer_8 * 16)) + ax3_inner_8) - 14400)] : pad_temp[(((ax2_8 * 122) + (ax3_outer_8 * 16)) + ax3_inner_8)]);
          }
        }
      }
    }
  }
  for (int32_t yy_16 = 0; yy_16 < 122; ++yy_16) {
    for (int32_t xx_outer_16 = 0; xx_outer_16 < 8; ++xx_outer_16) {
      for (int32_t xx_inner_16 = 0; xx_inner_16 < 16; ++xx_inner_16) {
        if (((xx_outer_16 * 8) + (xx_inner_16 >> 1)) < 61) {
          pad_temp[(((yy_16 * 122) + (xx_outer_16 * 16)) + xx_inner_16)] = 0.000000e+00f;
        }
        if (((xx_outer_16 * 8) + (xx_inner_16 >> 1)) < 61) {
          for (int32_t rc_5 = 0; rc_5 < 2; ++rc_5) {
            pad_temp[(((yy_16 * 122) + (xx_outer_16 * 16)) + xx_inner_16)] = (pad_temp[(((yy_16 * 122) + (xx_outer_16 * 16)) + xx_inner_16)] + (conv2d_nchw[((((rc_5 * 14884) + (yy_16 * 122)) + (xx_outer_16 * 16)) + xx_inner_16)] * ph_9[rc_5]));
          }
        }
      }
    }
  }
  for (int32_t i2_27 = 0; i2_27 < 122; ++i2_27) {
    for (int32_t i3_outer_27 = 0; i3_outer_27 < 8; ++i3_outer_27) {
      for (int32_t i3_inner_27 = 0; i3_inner_27 < 16; ++i3_inner_27) {
        if (((i3_outer_27 * 8) + (i3_inner_27 >> 1)) < 61) {
          conv2d_nchw[(((i2_27 * 122) + (i3_outer_27 * 16)) + i3_inner_27)] = max(pad_temp[(((i2_27 * 122) + (i3_outer_27 * 16)) + i3_inner_27)], 0.000000e+00f);
        }
      }
    }
  }
  for (int32_t i2_28 = 0; i2_28 < 122; ++i2_28) {
    for (int32_t i3_outer_28 = 0; i3_outer_28 < 8; ++i3_outer_28) {
      for (int32_t i3_inner_28 = 0; i3_inner_28 < 16; ++i3_inner_28) {
        if (((i3_outer_28 * 8) + (i3_inner_28 >> 1)) < 61) {
          pad_temp[(((i2_28 * 122) + (i3_outer_28 * 16)) + i3_inner_28)] = conv2d_nchw[(((i2_28 * 122) + (i3_outer_28 * 16)) + i3_inner_28)];
        }
      }
    }
  }
  for (int32_t yy_17 = 0; yy_17 < 122; ++yy_17) {
    for (int32_t xx_outer_17 = 0; xx_outer_17 < 8; ++xx_outer_17) {
      for (int32_t xx_inner_17 = 0; xx_inner_17 < 16; ++xx_inner_17) {
        if (((xx_outer_17 * 8) + (xx_inner_17 >> 1)) < 61) {
          conv2d_nchw[(((yy_17 * 122) + (xx_outer_17 * 16)) + xx_inner_17)] = 0.000000e+00f;
        }
        if (((xx_outer_17 * 8) + (xx_inner_17 >> 1)) < 61) {
          conv2d_nchw[(((yy_17 * 122) + (xx_outer_17 * 16)) + xx_inner_17)] = (conv2d_nchw[(((yy_17 * 122) + (xx_outer_17 * 16)) + xx_inner_17)] + (pad_temp[(((yy_17 * 122) + (xx_outer_17 * 16)) + xx_inner_17)] * ph_8[0]));
        }
      }
    }
  }
  for (int32_t i2_29 = 0; i2_29 < 122; ++i2_29) {
    for (int32_t i3_outer_29 = 0; i3_outer_29 < 8; ++i3_outer_29) {
      for (int32_t i3_inner_29 = 0; i3_inner_29 < 16; ++i3_inner_29) {
        if (((i3_outer_29 * 8) + (i3_inner_29 >> 1)) < 61) {
          pad_temp[(((i2_29 * 122) + (i3_outer_29 * 16)) + i3_inner_29)] = max(conv2d_nchw[(((i2_29 * 122) + (i3_outer_29 * 16)) + i3_inner_29)], 0.000000e+00f);
        }
      }
    }
  }
  for (int32_t i2_30 = 0; i2_30 < 122; ++i2_30) {
    for (int32_t i3_outer_30 = 0; i3_outer_30 < 8; ++i3_outer_30) {
      for (int32_t i3_inner_30 = 0; i3_inner_30 < 16; ++i3_inner_30) {
        if (((i3_outer_30 * 8) + (i3_inner_30 >> 1)) < 61) {
          conv2d_nchw[(((i2_30 * 122) + (i3_outer_30 * 16)) + i3_inner_30)] = pad_temp[(((i2_30 * 122) + (i3_outer_30 * 16)) + i3_inner_30)];
        }
      }
    }
  }
  for (int32_t yy_18 = 0; yy_18 < 120; ++yy_18) {
    for (int32_t xx_outer_18 = 0; xx_outer_18 < 8; ++xx_outer_18) {
      for (int32_t xx_inner_18 = 0; xx_inner_18 < 16; ++xx_inner_18) {
        if (((xx_outer_18 * 2) + (xx_inner_18 >> 3)) < 15) {
          conv2d_nchw_1[(((yy_18 * 120) + (xx_outer_18 * 16)) + xx_inner_18)] = 0.000000e+00f;
        }
        if (((xx_outer_18 * 2) + (xx_inner_18 >> 3)) < 15) {
          for (int32_t ry_6 = 0; ry_6 < 3; ++ry_6) {
            for (int32_t rx_6 = 0; rx_6 < 3; ++rx_6) {
              conv2d_nchw_1[(((yy_18 * 120) + (xx_outer_18 * 16)) + xx_inner_18)] = (conv2d_nchw_1[(((yy_18 * 120) + (xx_outer_18 * 16)) + xx_inner_18)] + (conv2d_nchw[(((((yy_18 * 122) + (ry_6 * 122)) + (xx_outer_18 * 16)) + xx_inner_18) + rx_6)] * ph_7[((ry_6 * 3) + rx_6)]));
            }
          }
        }
      }
    }
  }
  for (int32_t i2_31 = 0; i2_31 < 120; ++i2_31) {
    for (int32_t i3_outer_31 = 0; i3_outer_31 < 8; ++i3_outer_31) {
      for (int32_t i3_inner_31 = 0; i3_inner_31 < 16; ++i3_inner_31) {
        if (((i3_outer_31 * 2) + (i3_inner_31 >> 3)) < 15) {
          conv2d_nchw_1[(((i2_31 * 120) + (i3_outer_31 * 16)) + i3_inner_31)] = max(conv2d_nchw_1[(((i2_31 * 120) + (i3_outer_31 * 16)) + i3_inner_31)], 0.000000e+00f);
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_7 = 0; ax0_ax1_fused_7 < 2; ++ax0_ax1_fused_7) {
    for (int32_t ax2_9 = 0; ax2_9 < 122; ++ax2_9) {
      for (int32_t ax3_outer_9 = 0; ax3_outer_9 < 8; ++ax3_outer_9) {
        for (int32_t ax3_inner_9 = 0; ax3_inner_9 < 16; ++ax3_inner_9) {
          if (((ax3_outer_9 * 8) + (ax3_inner_9 >> 1)) < 61) {
            conv2d_nchw[((((ax0_ax1_fused_7 * 14884) + (ax2_9 * 122)) + (ax3_outer_9 * 16)) + ax3_inner_9)] = ((ax0_ax1_fused_7 == 1) ? conv2d_nchw_1[(((((((ax0_ax1_fused_7 + 1) >> 1) * 14400) + (ax2_9 * 120)) + (ax3_outer_9 * 16)) + ax3_inner_9) - 14400)] : pad_temp[(((ax2_9 * 122) + (ax3_outer_9 * 16)) + ax3_inner_9)]);
          }
        }
      }
    }
  }
  for (int32_t yy_19 = 0; yy_19 < 122; ++yy_19) {
    for (int32_t xx_outer_19 = 0; xx_outer_19 < 8; ++xx_outer_19) {
      for (int32_t xx_inner_19 = 0; xx_inner_19 < 16; ++xx_inner_19) {
        if (((xx_outer_19 * 8) + (xx_inner_19 >> 1)) < 61) {
          pad_temp[(((yy_19 * 122) + (xx_outer_19 * 16)) + xx_inner_19)] = 0.000000e+00f;
        }
        if (((xx_outer_19 * 8) + (xx_inner_19 >> 1)) < 61) {
          for (int32_t rc_6 = 0; rc_6 < 2; ++rc_6) {
            pad_temp[(((yy_19 * 122) + (xx_outer_19 * 16)) + xx_inner_19)] = (pad_temp[(((yy_19 * 122) + (xx_outer_19 * 16)) + xx_inner_19)] + (conv2d_nchw[((((rc_6 * 14884) + (yy_19 * 122)) + (xx_outer_19 * 16)) + xx_inner_19)] * ph_6[rc_6]));
          }
        }
      }
    }
  }
  for (int32_t i2_32 = 0; i2_32 < 122; ++i2_32) {
    for (int32_t i3_outer_32 = 0; i3_outer_32 < 8; ++i3_outer_32) {
      for (int32_t i3_inner_32 = 0; i3_inner_32 < 16; ++i3_inner_32) {
        if (((i3_outer_32 * 8) + (i3_inner_32 >> 1)) < 61) {
          conv2d_nchw[(((i2_32 * 122) + (i3_outer_32 * 16)) + i3_inner_32)] = max(pad_temp[(((i2_32 * 122) + (i3_outer_32 * 16)) + i3_inner_32)], 0.000000e+00f);
        }
      }
    }
  }
  for (int32_t i2_33 = 0; i2_33 < 122; ++i2_33) {
    for (int32_t i3_outer_33 = 0; i3_outer_33 < 8; ++i3_outer_33) {
      for (int32_t i3_inner_33 = 0; i3_inner_33 < 16; ++i3_inner_33) {
        if (((i3_outer_33 * 8) + (i3_inner_33 >> 1)) < 61) {
          pad_temp[(((i2_33 * 122) + (i3_outer_33 * 16)) + i3_inner_33)] = conv2d_nchw[(((i2_33 * 122) + (i3_outer_33 * 16)) + i3_inner_33)];
        }
      }
    }
  }
  for (int32_t yy_20 = 0; yy_20 < 122; ++yy_20) {
    for (int32_t xx_outer_20 = 0; xx_outer_20 < 8; ++xx_outer_20) {
      for (int32_t xx_inner_20 = 0; xx_inner_20 < 16; ++xx_inner_20) {
        if (((xx_outer_20 * 8) + (xx_inner_20 >> 1)) < 61) {
          conv2d_nchw[(((yy_20 * 122) + (xx_outer_20 * 16)) + xx_inner_20)] = 0.000000e+00f;
        }
        if (((xx_outer_20 * 8) + (xx_inner_20 >> 1)) < 61) {
          conv2d_nchw[(((yy_20 * 122) + (xx_outer_20 * 16)) + xx_inner_20)] = (conv2d_nchw[(((yy_20 * 122) + (xx_outer_20 * 16)) + xx_inner_20)] + (pad_temp[(((yy_20 * 122) + (xx_outer_20 * 16)) + xx_inner_20)] * ph_5[0]));
        }
      }
    }
  }
  for (int32_t i2_34 = 0; i2_34 < 122; ++i2_34) {
    for (int32_t i3_outer_34 = 0; i3_outer_34 < 8; ++i3_outer_34) {
      for (int32_t i3_inner_34 = 0; i3_inner_34 < 16; ++i3_inner_34) {
        if (((i3_outer_34 * 8) + (i3_inner_34 >> 1)) < 61) {
          pad_temp[(((i2_34 * 122) + (i3_outer_34 * 16)) + i3_inner_34)] = max(conv2d_nchw[(((i2_34 * 122) + (i3_outer_34 * 16)) + i3_inner_34)], 0.000000e+00f);
        }
      }
    }
  }
  for (int32_t i2_35 = 0; i2_35 < 122; ++i2_35) {
    for (int32_t i3_outer_35 = 0; i3_outer_35 < 8; ++i3_outer_35) {
      for (int32_t i3_inner_35 = 0; i3_inner_35 < 16; ++i3_inner_35) {
        if (((i3_outer_35 * 8) + (i3_inner_35 >> 1)) < 61) {
          conv2d_nchw[(((i2_35 * 122) + (i3_outer_35 * 16)) + i3_inner_35)] = pad_temp[(((i2_35 * 122) + (i3_outer_35 * 16)) + i3_inner_35)];
        }
      }
    }
  }
  for (int32_t yy_21 = 0; yy_21 < 120; ++yy_21) {
    for (int32_t xx_outer_21 = 0; xx_outer_21 < 8; ++xx_outer_21) {
      for (int32_t xx_inner_21 = 0; xx_inner_21 < 16; ++xx_inner_21) {
        if (((xx_outer_21 * 2) + (xx_inner_21 >> 3)) < 15) {
          conv2d_nchw_1[(((yy_21 * 120) + (xx_outer_21 * 16)) + xx_inner_21)] = 0.000000e+00f;
        }
        if (((xx_outer_21 * 2) + (xx_inner_21 >> 3)) < 15) {
          for (int32_t ry_7 = 0; ry_7 < 3; ++ry_7) {
            for (int32_t rx_7 = 0; rx_7 < 3; ++rx_7) {
              conv2d_nchw_1[(((yy_21 * 120) + (xx_outer_21 * 16)) + xx_inner_21)] = (conv2d_nchw_1[(((yy_21 * 120) + (xx_outer_21 * 16)) + xx_inner_21)] + (conv2d_nchw[(((((yy_21 * 122) + (ry_7 * 122)) + (xx_outer_21 * 16)) + xx_inner_21) + rx_7)] * ph_4[((ry_7 * 3) + rx_7)]));
            }
          }
        }
      }
    }
  }
  for (int32_t i2_36 = 0; i2_36 < 120; ++i2_36) {
    for (int32_t i3_outer_36 = 0; i3_outer_36 < 8; ++i3_outer_36) {
      for (int32_t i3_inner_36 = 0; i3_inner_36 < 16; ++i3_inner_36) {
        if (((i3_outer_36 * 2) + (i3_inner_36 >> 3)) < 15) {
          conv2d_nchw_1[(((i2_36 * 120) + (i3_outer_36 * 16)) + i3_inner_36)] = max(conv2d_nchw_1[(((i2_36 * 120) + (i3_outer_36 * 16)) + i3_inner_36)], 0.000000e+00f);
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_8 = 0; ax0_ax1_fused_8 < 2; ++ax0_ax1_fused_8) {
    for (int32_t ax2_10 = 0; ax2_10 < 122; ++ax2_10) {
      for (int32_t ax3_outer_10 = 0; ax3_outer_10 < 8; ++ax3_outer_10) {
        for (int32_t ax3_inner_10 = 0; ax3_inner_10 < 16; ++ax3_inner_10) {
          if (((ax3_outer_10 * 8) + (ax3_inner_10 >> 1)) < 61) {
            conv2d_nchw[((((ax0_ax1_fused_8 * 14884) + (ax2_10 * 122)) + (ax3_outer_10 * 16)) + ax3_inner_10)] = ((ax0_ax1_fused_8 == 1) ? conv2d_nchw_1[(((((((ax0_ax1_fused_8 + 1) >> 1) * 14400) + (ax2_10 * 120)) + (ax3_outer_10 * 16)) + ax3_inner_10) - 14400)] : pad_temp[(((ax2_10 * 122) + (ax3_outer_10 * 16)) + ax3_inner_10)]);
          }
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_9 = 0; ax0_ax1_fused_9 < 2; ++ax0_ax1_fused_9) {
    for (int32_t ax2_11 = 0; ax2_11 < 124; ++ax2_11) {
      for (int32_t ax3_outer_11 = 0; ax3_outer_11 < 8; ++ax3_outer_11) {
        for (int32_t ax3_inner_11 = 0; ax3_inner_11 < 16; ++ax3_inner_11) {
          if (((ax3_outer_11 * 4) + (ax3_inner_11 >> 2)) < 31) {
            pad_temp[((((ax0_ax1_fused_9 * 15376) + (ax2_11 * 124)) + (ax3_outer_11 * 16)) + ax3_inner_11)] = (((((1 <= ax2_11) && (ax2_11 < 123)) && (1 <= ((ax3_outer_11 * 16) + ax3_inner_11))) && (((ax3_outer_11 * 16) + ax3_inner_11) < 123)) ? conv2d_nchw[(((((ax0_ax1_fused_9 * 14884) + (ax2_11 * 122)) + (ax3_outer_11 * 16)) + ax3_inner_11) - 123)] : -3.402823e+38f);
          }
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_10 = 0; ax0_ax1_fused_10 < 2; ++ax0_ax1_fused_10) {
    for (int32_t ax2_12 = 0; ax2_12 < 122; ++ax2_12) {
      for (int32_t ax3_outer_12 = 0; ax3_outer_12 < 8; ++ax3_outer_12) {
        for (int32_t ax3_inner_12 = 0; ax3_inner_12 < 16; ++ax3_inner_12) {
          if (((ax3_outer_12 * 8) + (ax3_inner_12 >> 1)) < 61) {
            conv2d_nchw[((((ax0_ax1_fused_10 * 14884) + (ax2_12 * 122)) + (ax3_outer_12 * 16)) + ax3_inner_12)] = -3.402823e+38f;
          }
          if (((ax3_outer_12 * 8) + (ax3_inner_12 >> 1)) < 61) {
            for (int32_t rv0_2 = 0; rv0_2 < 3; ++rv0_2) {
              for (int32_t rv1_2 = 0; rv1_2 < 3; ++rv1_2) {
                conv2d_nchw[((((ax0_ax1_fused_10 * 14884) + (ax2_12 * 122)) + (ax3_outer_12 * 16)) + ax3_inner_12)] = max(conv2d_nchw[((((ax0_ax1_fused_10 * 14884) + (ax2_12 * 122)) + (ax3_outer_12 * 16)) + ax3_inner_12)], pad_temp[((((((ax0_ax1_fused_10 * 15376) + (ax2_12 * 124)) + (rv0_2 * 124)) + (ax3_outer_12 * 16)) + ax3_inner_12) + rv1_2)]);
              }
            }
          }
        }
      }
    }
  }
  for (int32_t yy_22 = 0; yy_22 < 122; ++yy_22) {
    for (int32_t xx_outer_22 = 0; xx_outer_22 < 8; ++xx_outer_22) {
      for (int32_t xx_inner_22 = 0; xx_inner_22 < 16; ++xx_inner_22) {
        if (((xx_outer_22 * 8) + (xx_inner_22 >> 1)) < 61) {
          pad_temp[(((yy_22 * 122) + (xx_outer_22 * 16)) + xx_inner_22)] = 0.000000e+00f;
        }
        if (((xx_outer_22 * 8) + (xx_inner_22 >> 1)) < 61) {
          for (int32_t rc_7 = 0; rc_7 < 2; ++rc_7) {
            pad_temp[(((yy_22 * 122) + (xx_outer_22 * 16)) + xx_inner_22)] = (pad_temp[(((yy_22 * 122) + (xx_outer_22 * 16)) + xx_inner_22)] + (conv2d_nchw[((((rc_7 * 14884) + (yy_22 * 122)) + (xx_outer_22 * 16)) + xx_inner_22)] * ph_3[rc_7]));
          }
        }
      }
    }
  }
  for (int32_t i2_37 = 0; i2_37 < 122; ++i2_37) {
    for (int32_t i3_outer_37 = 0; i3_outer_37 < 8; ++i3_outer_37) {
      for (int32_t i3_inner_37 = 0; i3_inner_37 < 16; ++i3_inner_37) {
        if (((i3_outer_37 * 8) + (i3_inner_37 >> 1)) < 61) {
          conv2d_nchw[(((i2_37 * 122) + (i3_outer_37 * 16)) + i3_inner_37)] = max(pad_temp[(((i2_37 * 122) + (i3_outer_37 * 16)) + i3_inner_37)], 0.000000e+00f);
        }
      }
    }
  }
  for (int32_t i2_38 = 0; i2_38 < 122; ++i2_38) {
    for (int32_t i3_outer_38 = 0; i3_outer_38 < 8; ++i3_outer_38) {
      for (int32_t i3_inner_38 = 0; i3_inner_38 < 16; ++i3_inner_38) {
        if (((i3_outer_38 * 8) + (i3_inner_38 >> 1)) < 61) {
          pad_temp[(((i2_38 * 122) + (i3_outer_38 * 16)) + i3_inner_38)] = conv2d_nchw[(((i2_38 * 122) + (i3_outer_38 * 16)) + i3_inner_38)];
        }
      }
    }
  }
  for (int32_t yy_23 = 0; yy_23 < 122; ++yy_23) {
    for (int32_t xx_outer_23 = 0; xx_outer_23 < 8; ++xx_outer_23) {
      for (int32_t xx_inner_23 = 0; xx_inner_23 < 16; ++xx_inner_23) {
        if (((xx_outer_23 * 8) + (xx_inner_23 >> 1)) < 61) {
          conv2d_nchw[(((yy_23 * 122) + (xx_outer_23 * 16)) + xx_inner_23)] = 0.000000e+00f;
        }
        if (((xx_outer_23 * 8) + (xx_inner_23 >> 1)) < 61) {
          conv2d_nchw[(((yy_23 * 122) + (xx_outer_23 * 16)) + xx_inner_23)] = (conv2d_nchw[(((yy_23 * 122) + (xx_outer_23 * 16)) + xx_inner_23)] + (pad_temp[(((yy_23 * 122) + (xx_outer_23 * 16)) + xx_inner_23)] * ph_2[0]));
        }
      }
    }
  }
  for (int32_t i2_39 = 0; i2_39 < 122; ++i2_39) {
    for (int32_t i3_outer_39 = 0; i3_outer_39 < 8; ++i3_outer_39) {
      for (int32_t i3_inner_39 = 0; i3_inner_39 < 16; ++i3_inner_39) {
        if (((i3_outer_39 * 8) + (i3_inner_39 >> 1)) < 61) {
          pad_temp[(((i2_39 * 122) + (i3_outer_39 * 16)) + i3_inner_39)] = max(conv2d_nchw[(((i2_39 * 122) + (i3_outer_39 * 16)) + i3_inner_39)], 0.000000e+00f);
        }
      }
    }
  }
  for (int32_t i2_40 = 0; i2_40 < 122; ++i2_40) {
    for (int32_t i3_outer_40 = 0; i3_outer_40 < 8; ++i3_outer_40) {
      for (int32_t i3_inner_40 = 0; i3_inner_40 < 16; ++i3_inner_40) {
        if (((i3_outer_40 * 8) + (i3_inner_40 >> 1)) < 61) {
          conv2d_nchw[(((i2_40 * 122) + (i3_outer_40 * 16)) + i3_inner_40)] = pad_temp[(((i2_40 * 122) + (i3_outer_40 * 16)) + i3_inner_40)];
        }
      }
    }
  }
  for (int32_t yy_24 = 0; yy_24 < 120; ++yy_24) {
    for (int32_t xx_outer_24 = 0; xx_outer_24 < 8; ++xx_outer_24) {
      for (int32_t xx_inner_24 = 0; xx_inner_24 < 16; ++xx_inner_24) {
        if (((xx_outer_24 * 2) + (xx_inner_24 >> 3)) < 15) {
          conv2d_nchw_1[(((yy_24 * 120) + (xx_outer_24 * 16)) + xx_inner_24)] = 0.000000e+00f;
        }
        if (((xx_outer_24 * 2) + (xx_inner_24 >> 3)) < 15) {
          for (int32_t ry_8 = 0; ry_8 < 3; ++ry_8) {
            for (int32_t rx_8 = 0; rx_8 < 3; ++rx_8) {
              conv2d_nchw_1[(((yy_24 * 120) + (xx_outer_24 * 16)) + xx_inner_24)] = (conv2d_nchw_1[(((yy_24 * 120) + (xx_outer_24 * 16)) + xx_inner_24)] + (conv2d_nchw[(((((yy_24 * 122) + (ry_8 * 122)) + (xx_outer_24 * 16)) + xx_inner_24) + rx_8)] * ph_1[((ry_8 * 3) + rx_8)]));
            }
          }
        }
      }
    }
  }
  for (int32_t i2_41 = 0; i2_41 < 120; ++i2_41) {
    for (int32_t i3_outer_41 = 0; i3_outer_41 < 8; ++i3_outer_41) {
      for (int32_t i3_inner_41 = 0; i3_inner_41 < 16; ++i3_inner_41) {
        if (((i3_outer_41 * 2) + (i3_inner_41 >> 3)) < 15) {
          conv2d_nchw_1[(((i2_41 * 120) + (i3_outer_41 * 16)) + i3_inner_41)] = max(conv2d_nchw_1[(((i2_41 * 120) + (i3_outer_41 * 16)) + i3_inner_41)], 0.000000e+00f);
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_11 = 0; ax0_ax1_fused_11 < 2; ++ax0_ax1_fused_11) {
    for (int32_t ax2_13 = 0; ax2_13 < 122; ++ax2_13) {
      for (int32_t ax3_outer_13 = 0; ax3_outer_13 < 8; ++ax3_outer_13) {
        for (int32_t ax3_inner_13 = 0; ax3_inner_13 < 16; ++ax3_inner_13) {
          if (((ax3_outer_13 * 8) + (ax3_inner_13 >> 1)) < 61) {
            T_concat[((((ax0_ax1_fused_11 * 14884) + (ax2_13 * 122)) + (ax3_outer_13 * 16)) + ax3_inner_13)] = ((ax0_ax1_fused_11 == 1) ? conv2d_nchw_1[(((((((ax0_ax1_fused_11 + 1) >> 1) * 14400) + (ax2_13 * 120)) + (ax3_outer_13 * 16)) + ax3_inner_13) - 14400)] : pad_temp[(((ax2_13 * 122) + (ax3_outer_13 * 16)) + ax3_inner_13)]);
          }
        }
      }
    }
  }
}

