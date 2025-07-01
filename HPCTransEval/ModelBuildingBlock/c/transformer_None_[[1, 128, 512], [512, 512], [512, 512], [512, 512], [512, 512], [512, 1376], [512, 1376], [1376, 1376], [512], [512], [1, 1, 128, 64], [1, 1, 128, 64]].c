void default_function_kernel(float* T_add, float* ph, float* ph_1, float* ph_2, float* ph_3, float* ph_4, float* ph_5, float* ph_6, float* ph_7, float* ph_8, float* ph_9, float* ph_10, float* ph_11) {
  float T_cast[65536];
  float T_cast_1[1024];
  float T_multiply[66048];
  float T_multiply_red[128];
  float T_matmul_NN[65536];
  float T_transpose[65536];
  float T_strided_slice[32768];
  float T_strided_slice_1[131072];
  float T_strided_slice_2[32768];
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 128; ++ax0_ax1_fused) {
    for (int32_t ax2_outer = 0; ax2_outer < 32; ++ax2_outer) {
      for (int32_t ax2_inner = 0; ax2_inner < 16; ++ax2_inner) {
        T_cast[(((ax0_ax1_fused * 512) + (ax2_outer * 16)) + ax2_inner)] = ph[(((ax0_ax1_fused * 512) + (ax2_outer * 16)) + ax2_inner)];
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_outer = 0; ax0_outer < 32; ++ax0_outer) {
    for (int32_t ax0_inner = 0; ax0_inner < 16; ++ax0_inner) {
      T_cast_1[((ax0_outer * 16) + ax0_inner)] = ph_1[((ax0_outer * 16) + ax0_inner)];
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < 128; ++ax0_ax1_fused_1) {
    for (int32_t ax2_outer_1 = 0; ax2_outer_1 < 32; ++ax2_outer_1) {
      for (int32_t ax2_inner_1 = 0; ax2_inner_1 < 16; ++ax2_inner_1) {
        T_multiply[(((ax0_ax1_fused_1 * 512) + (ax2_outer_1 * 16)) + ax2_inner_1)] = (T_cast[(((ax0_ax1_fused_1 * 512) + (ax2_outer_1 * 16)) + ax2_inner_1)] * T_cast[(((ax0_ax1_fused_1 * 512) + (ax2_outer_1 * 16)) + ax2_inner_1)]);
      }
    }
  }
  for (int32_t ax1_outer = 0; ax1_outer < 8; ++ax1_outer) {
    for (int32_t ax1_inner = 0; ax1_inner < 16; ++ax1_inner) {
      T_multiply_red[((ax1_outer * 16) + ax1_inner)] = 0.000000e+00f;
      for (int32_t k2 = 0; k2 < 512; ++k2) {
        T_multiply_red[((ax1_outer * 16) + ax1_inner)] = (T_multiply_red[((ax1_outer * 16) + ax1_inner)] + T_multiply[(((ax1_outer * 8192) + (ax1_inner * 512)) + k2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_2 = 0; ax0_ax1_fused_2 < 128; ++ax0_ax1_fused_2) {
    for (int32_t ax2_outer_2 = 0; ax2_outer_2 < 32; ++ax2_outer_2) {
      for (int32_t ax2_inner_2 = 0; ax2_inner_2 < 16; ++ax2_inner_2) {
        T_cast[(((ax0_ax1_fused_2 * 512) + (ax2_outer_2 * 16)) + ax2_inner_2)] = ((T_cast[(((ax0_ax1_fused_2 * 512) + (ax2_outer_2 * 16)) + ax2_inner_2)] * T_cast_1[((ax2_outer_2 * 16) + ax2_inner_2)]) * (1.000000e+00f / sqrtf(((T_multiply_red[ax0_ax1_fused_2] * 1.953125e-03f) + 1.000000e-05f))));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 128; ++ax0) {
    for (int32_t ax1_outer_1 = 0; ax1_outer_1 < 32; ++ax1_outer_1) {
      for (int32_t ax1_inner_1 = 0; ax1_inner_1 < 16; ++ax1_inner_1) {
        T_multiply[(((ax0 * 512) + (ax1_outer_1 * 16)) + ax1_inner_1)] = T_cast[(((ax0 * 512) + (ax1_outer_1 * 16)) + ax1_inner_1)];
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 128; ++i0) {
    for (int32_t i1_outer = 0; i1_outer < 32; ++i1_outer) {
      for (int32_t i1_inner = 0; i1_inner < 16; ++i1_inner) {
        T_matmul_NN[(((i0 * 512) + (i1_outer * 16)) + i1_inner)] = 0.000000e+00f;
        for (int32_t k = 0; k < 512; ++k) {
          T_matmul_NN[(((i0 * 512) + (i1_outer * 16)) + i1_inner)] = (T_matmul_NN[(((i0 * 512) + (i1_outer * 16)) + i1_inner)] + (T_multiply[((i0 * 512) + k)] * ph_2[(((k * 512) + (i1_outer * 16)) + i1_inner)]));
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_3 = 0; ax0_ax1_fused_3 < 8; ++ax0_ax1_fused_3) {
    for (int32_t ax2 = 0; ax2 < 128; ++ax2) {
      for (int32_t ax3_outer = 0; ax3_outer < 4; ++ax3_outer) {
        for (int32_t ax3_inner = 0; ax3_inner < 16; ++ax3_inner) {
          T_transpose[((((ax0_ax1_fused_3 * 8192) + (ax2 * 64)) + (ax3_outer * 16)) + ax3_inner)] = T_matmul_NN[((((ax2 * 512) + (ax0_ax1_fused_3 * 64)) + (ax3_outer * 16)) + ax3_inner)];
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_4 = 0; ax0_ax1_fused_4 < 8; ++ax0_ax1_fused_4) {
    for (int32_t ax2_1 = 0; ax2_1 < 128; ++ax2_1) {
      for (int32_t ax3_outer_1 = 0; ax3_outer_1 < 4; ++ax3_outer_1) {
        for (int32_t ax3_inner_1 = 0; ax3_inner_1 < 16; ++ax3_inner_1) {
          T_matmul_NN[((((ax0_ax1_fused_4 * 8192) + (ax2_1 * 64)) + (ax3_outer_1 * 16)) + ax3_inner_1)] = (T_transpose[((((ax0_ax1_fused_4 * 8192) + (ax2_1 * 64)) + (ax3_outer_1 * 16)) + ax3_inner_1)] * ph_3[(((ax2_1 * 64) + (ax3_outer_1 * 16)) + ax3_inner_1)]);
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_5 = 0; ax0_ax1_fused_5 < 8; ++ax0_ax1_fused_5) {
    for (int32_t ax2_2 = 0; ax2_2 < 128; ++ax2_2) {
      for (int32_t ax3_outer_2 = 0; ax3_outer_2 < 2; ++ax3_outer_2) {
        for (int32_t ax3_inner_2 = 0; ax3_inner_2 < 16; ++ax3_inner_2) {
          T_strided_slice[((((ax0_ax1_fused_5 * 4096) + (ax2_2 * 32)) + (ax3_outer_2 * 16)) + ax3_inner_2)] = T_transpose[(((((ax0_ax1_fused_5 * 8192) + (ax2_2 * 64)) + (ax3_outer_2 * 16)) + ax3_inner_2) + 32)];
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_6 = 0; ax0_ax1_fused_6 < 8; ++ax0_ax1_fused_6) {
    for (int32_t ax2_3 = 0; ax2_3 < 128; ++ax2_3) {
      for (int32_t ax3_outer_3 = 0; ax3_outer_3 < 2; ++ax3_outer_3) {
        for (int32_t ax3_inner_3 = 0; ax3_inner_3 < 16; ++ax3_inner_3) {
          T_strided_slice_1[((((ax0_ax1_fused_6 * 4096) + (ax2_3 * 32)) + (ax3_outer_3 * 16)) + ax3_inner_3)] = T_transpose[((((ax0_ax1_fused_6 * 8192) + (ax2_3 * 64)) + (ax3_outer_3 * 16)) + ax3_inner_3)];
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_7 = 0; ax0_ax1_fused_7 < 8; ++ax0_ax1_fused_7) {
    for (int32_t ax2_4 = 0; ax2_4 < 128; ++ax2_4) {
      for (int32_t ax3_outer_4 = 0; ax3_outer_4 < 4; ++ax3_outer_4) {
        for (int32_t ax3_inner_4 = 0; ax3_inner_4 < 16; ++ax3_inner_4) {
          T_transpose[((((ax0_ax1_fused_7 * 8192) + (ax2_4 * 64)) + (ax3_outer_4 * 16)) + ax3_inner_4)] = ((2 <= ax3_outer_4) ? T_strided_slice[(((((ax0_ax1_fused_7 * 4096) + (ax2_4 * 32)) + (ax3_outer_4 * 16)) + ax3_inner_4) - 32)] : T_strided_slice_1[((((ax0_ax1_fused_7 * 4096) + (ax2_4 * 32)) + (ax3_outer_4 * 16)) + ax3_inner_4)]);
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_8 = 0; ax0_ax1_fused_8 < 8; ++ax0_ax1_fused_8) {
    for (int32_t ax2_5 = 0; ax2_5 < 128; ++ax2_5) {
      for (int32_t ax3_outer_5 = 0; ax3_outer_5 < 4; ++ax3_outer_5) {
        for (int32_t ax3_inner_5 = 0; ax3_inner_5 < 16; ++ax3_inner_5) {
          T_transpose[((((ax0_ax1_fused_8 * 8192) + (ax2_5 * 64)) + (ax3_outer_5 * 16)) + ax3_inner_5)] = (T_transpose[((((ax0_ax1_fused_8 * 8192) + (ax2_5 * 64)) + (ax3_outer_5 * 16)) + ax3_inner_5)] * ph_4[(((ax2_5 * 64) + (ax3_outer_5 * 16)) + ax3_inner_5)]);
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_9 = 0; ax0_ax1_fused_9 < 8; ++ax0_ax1_fused_9) {
    for (int32_t ax2_6 = 0; ax2_6 < 128; ++ax2_6) {
      for (int32_t ax3_outer_6 = 0; ax3_outer_6 < 4; ++ax3_outer_6) {
        for (int32_t ax3_inner_6 = 0; ax3_inner_6 < 16; ++ax3_inner_6) {
          T_matmul_NN[((((ax0_ax1_fused_9 * 8192) + (ax2_6 * 64)) + (ax3_outer_6 * 16)) + ax3_inner_6)] = (T_matmul_NN[((((ax0_ax1_fused_9 * 8192) + (ax2_6 * 64)) + (ax3_outer_6 * 16)) + ax3_inner_6)] + T_transpose[((((ax0_ax1_fused_9 * 8192) + (ax2_6 * 64)) + (ax3_outer_6 * 16)) + ax3_inner_6)]);
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_1 = 0; i0_1 < 128; ++i0_1) {
    for (int32_t i1_outer_1 = 0; i1_outer_1 < 32; ++i1_outer_1) {
      for (int32_t i1_inner_1 = 0; i1_inner_1 < 16; ++i1_inner_1) {
        T_transpose[(((i0_1 * 512) + (i1_outer_1 * 16)) + i1_inner_1)] = 0.000000e+00f;
        for (int32_t k_1 = 0; k_1 < 512; ++k_1) {
          T_transpose[(((i0_1 * 512) + (i1_outer_1 * 16)) + i1_inner_1)] = (T_transpose[(((i0_1 * 512) + (i1_outer_1 * 16)) + i1_inner_1)] + (T_multiply[((i0_1 * 512) + k_1)] * ph_5[(((k_1 * 512) + (i1_outer_1 * 16)) + i1_inner_1)]));
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_10 = 0; ax0_ax1_fused_10 < 8; ++ax0_ax1_fused_10) {
    for (int32_t ax2_7 = 0; ax2_7 < 128; ++ax2_7) {
      for (int32_t ax3_outer_7 = 0; ax3_outer_7 < 4; ++ax3_outer_7) {
        for (int32_t ax3_inner_7 = 0; ax3_inner_7 < 16; ++ax3_inner_7) {
          T_strided_slice_1[((((ax0_ax1_fused_10 * 8192) + (ax2_7 * 64)) + (ax3_outer_7 * 16)) + ax3_inner_7)] = T_transpose[((((ax2_7 * 512) + (ax0_ax1_fused_10 * 64)) + (ax3_outer_7 * 16)) + ax3_inner_7)];
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_11 = 0; ax0_ax1_fused_11 < 8; ++ax0_ax1_fused_11) {
    for (int32_t ax2_8 = 0; ax2_8 < 128; ++ax2_8) {
      for (int32_t ax3_outer_8 = 0; ax3_outer_8 < 4; ++ax3_outer_8) {
        for (int32_t ax3_inner_8 = 0; ax3_inner_8 < 16; ++ax3_inner_8) {
          T_transpose[((((ax0_ax1_fused_11 * 8192) + (ax2_8 * 64)) + (ax3_outer_8 * 16)) + ax3_inner_8)] = (T_strided_slice_1[((((ax0_ax1_fused_11 * 8192) + (ax2_8 * 64)) + (ax3_outer_8 * 16)) + ax3_inner_8)] * ph_3[(((ax2_8 * 64) + (ax3_outer_8 * 16)) + ax3_inner_8)]);
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_12 = 0; ax0_ax1_fused_12 < 8; ++ax0_ax1_fused_12) {
    for (int32_t ax2_9 = 0; ax2_9 < 128; ++ax2_9) {
      for (int32_t ax3_outer_9 = 0; ax3_outer_9 < 2; ++ax3_outer_9) {
        for (int32_t ax3_inner_9 = 0; ax3_inner_9 < 16; ++ax3_inner_9) {
          T_strided_slice[((((ax0_ax1_fused_12 * 4096) + (ax2_9 * 32)) + (ax3_outer_9 * 16)) + ax3_inner_9)] = T_strided_slice_1[(((((ax0_ax1_fused_12 * 8192) + (ax2_9 * 64)) + (ax3_outer_9 * 16)) + ax3_inner_9) + 32)];
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_13 = 0; ax0_ax1_fused_13 < 8; ++ax0_ax1_fused_13) {
    for (int32_t ax2_10 = 0; ax2_10 < 128; ++ax2_10) {
      for (int32_t ax3_outer_10 = 0; ax3_outer_10 < 2; ++ax3_outer_10) {
        for (int32_t ax3_inner_10 = 0; ax3_inner_10 < 16; ++ax3_inner_10) {
          T_strided_slice_2[((((ax0_ax1_fused_13 * 4096) + (ax2_10 * 32)) + (ax3_outer_10 * 16)) + ax3_inner_10)] = T_strided_slice_1[((((ax0_ax1_fused_13 * 8192) + (ax2_10 * 64)) + (ax3_outer_10 * 16)) + ax3_inner_10)];
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_14 = 0; ax0_ax1_fused_14 < 8; ++ax0_ax1_fused_14) {
    for (int32_t ax2_11 = 0; ax2_11 < 128; ++ax2_11) {
      for (int32_t ax3_outer_11 = 0; ax3_outer_11 < 4; ++ax3_outer_11) {
        for (int32_t ax3_inner_11 = 0; ax3_inner_11 < 16; ++ax3_inner_11) {
          T_strided_slice_1[((((ax0_ax1_fused_14 * 8192) + (ax2_11 * 64)) + (ax3_outer_11 * 16)) + ax3_inner_11)] = ((2 <= ax3_outer_11) ? T_strided_slice[(((((ax0_ax1_fused_14 * 4096) + (ax2_11 * 32)) + (ax3_outer_11 * 16)) + ax3_inner_11) - 32)] : T_strided_slice_2[((((ax0_ax1_fused_14 * 4096) + (ax2_11 * 32)) + (ax3_outer_11 * 16)) + ax3_inner_11)]);
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_15 = 0; ax0_ax1_fused_15 < 8; ++ax0_ax1_fused_15) {
    for (int32_t ax2_12 = 0; ax2_12 < 128; ++ax2_12) {
      for (int32_t ax3_outer_12 = 0; ax3_outer_12 < 4; ++ax3_outer_12) {
        for (int32_t ax3_inner_12 = 0; ax3_inner_12 < 16; ++ax3_inner_12) {
          T_strided_slice_1[((((ax0_ax1_fused_15 * 8192) + (ax2_12 * 64)) + (ax3_outer_12 * 16)) + ax3_inner_12)] = (T_strided_slice_1[((((ax0_ax1_fused_15 * 8192) + (ax2_12 * 64)) + (ax3_outer_12 * 16)) + ax3_inner_12)] * ph_4[(((ax2_12 * 64) + (ax3_outer_12 * 16)) + ax3_inner_12)]);
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_16 = 0; ax0_ax1_fused_16 < 8; ++ax0_ax1_fused_16) {
    for (int32_t ax2_13 = 0; ax2_13 < 128; ++ax2_13) {
      for (int32_t ax3_outer_13 = 0; ax3_outer_13 < 4; ++ax3_outer_13) {
        for (int32_t ax3_inner_13 = 0; ax3_inner_13 < 16; ++ax3_inner_13) {
          T_transpose[((((ax0_ax1_fused_16 * 8192) + (ax2_13 * 64)) + (ax3_outer_13 * 16)) + ax3_inner_13)] = (T_transpose[((((ax0_ax1_fused_16 * 8192) + (ax2_13 * 64)) + (ax3_outer_13 * 16)) + ax3_inner_13)] + T_strided_slice_1[((((ax0_ax1_fused_16 * 8192) + (ax2_13 * 64)) + (ax3_outer_13 * 16)) + ax3_inner_13)]);
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_17 = 0; ax0_ax1_fused_17 < 8; ++ax0_ax1_fused_17) {
    for (int32_t ax2_14 = 0; ax2_14 < 128; ++ax2_14) {
      for (int32_t ax3_outer_14 = 0; ax3_outer_14 < 8; ++ax3_outer_14) {
        for (int32_t ax3_inner_14 = 0; ax3_inner_14 < 16; ++ax3_inner_14) {
          T_strided_slice_1[((((ax0_ax1_fused_17 * 16384) + (ax2_14 * 128)) + (ax3_outer_14 * 16)) + ax3_inner_14)] = 0.000000e+00f;
          for (int32_t h = 0; h < 64; ++h) {
            T_strided_slice_1[((((ax0_ax1_fused_17 * 16384) + (ax2_14 * 128)) + (ax3_outer_14 * 16)) + ax3_inner_14)] = (T_strided_slice_1[((((ax0_ax1_fused_17 * 16384) + (ax2_14 * 128)) + (ax3_outer_14 * 16)) + ax3_inner_14)] + (T_matmul_NN[(((ax0_ax1_fused_17 * 8192) + (ax2_14 * 64)) + h)] * T_transpose[((((ax0_ax1_fused_17 * 8192) + (ax3_outer_14 * 1024)) + (ax3_inner_14 * 64)) + h)]));
          }
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 8; ++i0_i1_fused) {
    for (int32_t i2_outer = 0; i2_outer < 8; ++i2_outer) {
      for (int32_t i2_inner = 0; i2_inner < 16; ++i2_inner) {
        T_cast_1[(((i0_i1_fused * 128) + (i2_outer * 16)) + i2_inner)] = -3.402823e+38f;
        for (int32_t k_2 = 0; k_2 < 128; ++k_2) {
          T_cast_1[(((i0_i1_fused * 128) + (i2_outer * 16)) + i2_inner)] = max(T_cast_1[(((i0_i1_fused * 128) + (i2_outer * 16)) + i2_inner)], T_strided_slice_1[((((i0_i1_fused * 16384) + (i2_outer * 2048)) + (i2_inner * 128)) + k_2)]);
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_1 = 0; i0_i1_fused_1 < 8; ++i0_i1_fused_1) {
    for (int32_t i2 = 0; i2 < 128; ++i2) {
      for (int32_t i3_outer = 0; i3_outer < 8; ++i3_outer) {
        for (int32_t i3_inner = 0; i3_inner < 16; ++i3_inner) {
          T_strided_slice_1[((((i0_i1_fused_1 * 16384) + (i2 * 128)) + (i3_outer * 16)) + i3_inner)] = expf((T_strided_slice_1[((((i0_i1_fused_1 * 16384) + (i2 * 128)) + (i3_outer * 16)) + i3_inner)] - T_cast_1[((i0_i1_fused_1 * 128) + i2)]));
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_2 = 0; i0_i1_fused_2 < 8; ++i0_i1_fused_2) {
    for (int32_t i2_outer_1 = 0; i2_outer_1 < 8; ++i2_outer_1) {
      for (int32_t i2_inner_1 = 0; i2_inner_1 < 16; ++i2_inner_1) {
        T_cast_1[(((i0_i1_fused_2 * 128) + (i2_outer_1 * 16)) + i2_inner_1)] = 0.000000e+00f;
        for (int32_t k_3 = 0; k_3 < 128; ++k_3) {
          T_cast_1[(((i0_i1_fused_2 * 128) + (i2_outer_1 * 16)) + i2_inner_1)] = (T_cast_1[(((i0_i1_fused_2 * 128) + (i2_outer_1 * 16)) + i2_inner_1)] + T_strided_slice_1[((((i0_i1_fused_2 * 16384) + (i2_outer_1 * 2048)) + (i2_inner_1 * 128)) + k_3)]);
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused_3 = 0; i0_i1_fused_3 < 8; ++i0_i1_fused_3) {
    for (int32_t i2_1 = 0; i2_1 < 128; ++i2_1) {
      for (int32_t i3_outer_1 = 0; i3_outer_1 < 8; ++i3_outer_1) {
        for (int32_t i3_inner_1 = 0; i3_inner_1 < 16; ++i3_inner_1) {
          T_strided_slice_1[((((i0_i1_fused_3 * 16384) + (i2_1 * 128)) + (i3_outer_1 * 16)) + i3_inner_1)] = (T_strided_slice_1[((((i0_i1_fused_3 * 16384) + (i2_1 * 128)) + (i3_outer_1 * 16)) + i3_inner_1)] / T_cast_1[((i0_i1_fused_3 * 128) + i2_1)]);
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_2 = 0; i0_2 < 128; ++i0_2) {
    for (int32_t i1_outer_2 = 0; i1_outer_2 < 32; ++i1_outer_2) {
      for (int32_t i1_inner_2 = 0; i1_inner_2 < 16; ++i1_inner_2) {
        T_matmul_NN[(((i0_2 * 512) + (i1_outer_2 * 16)) + i1_inner_2)] = 0.000000e+00f;
        for (int32_t k_4 = 0; k_4 < 512; ++k_4) {
          T_matmul_NN[(((i0_2 * 512) + (i1_outer_2 * 16)) + i1_inner_2)] = (T_matmul_NN[(((i0_2 * 512) + (i1_outer_2 * 16)) + i1_inner_2)] + (T_multiply[((i0_2 * 512) + k_4)] * ph_6[(((k_4 * 512) + (i1_outer_2 * 16)) + i1_inner_2)]));
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_18 = 0; ax0_ax1_fused_18 < 8; ++ax0_ax1_fused_18) {
    for (int32_t ax2_15 = 0; ax2_15 < 128; ++ax2_15) {
      for (int32_t ax3_outer_15 = 0; ax3_outer_15 < 4; ++ax3_outer_15) {
        for (int32_t ax3_inner_15 = 0; ax3_inner_15 < 16; ++ax3_inner_15) {
          T_transpose[((((ax0_ax1_fused_18 * 8192) + (ax2_15 * 64)) + (ax3_outer_15 * 16)) + ax3_inner_15)] = T_matmul_NN[((((ax2_15 * 512) + (ax0_ax1_fused_18 * 64)) + (ax3_outer_15 * 16)) + ax3_inner_15)];
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_19 = 0; ax0_ax1_fused_19 < 8; ++ax0_ax1_fused_19) {
    for (int32_t ax2_16 = 0; ax2_16 < 128; ++ax2_16) {
      for (int32_t ax3_outer_16 = 0; ax3_outer_16 < 4; ++ax3_outer_16) {
        for (int32_t ax3_inner_16 = 0; ax3_inner_16 < 16; ++ax3_inner_16) {
          T_multiply[((((ax0_ax1_fused_19 * 8192) + (ax2_16 * 64)) + (ax3_outer_16 * 16)) + ax3_inner_16)] = 0.000000e+00f;
          for (int32_t k_5 = 0; k_5 < 128; ++k_5) {
            T_multiply[((((ax0_ax1_fused_19 * 8192) + (ax2_16 * 64)) + (ax3_outer_16 * 16)) + ax3_inner_16)] = (T_multiply[((((ax0_ax1_fused_19 * 8192) + (ax2_16 * 64)) + (ax3_outer_16 * 16)) + ax3_inner_16)] + (T_strided_slice_1[(((ax0_ax1_fused_19 * 16384) + (ax2_16 * 128)) + k_5)] * T_transpose[((((ax0_ax1_fused_19 * 8192) + (k_5 * 64)) + (ax3_outer_16 * 16)) + ax3_inner_16)]));
          }
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_20 = 0; ax0_ax1_fused_20 < 128; ++ax0_ax1_fused_20) {
    for (int32_t ax2_17 = 0; ax2_17 < 8; ++ax2_17) {
      for (int32_t ax3_outer_17 = 0; ax3_outer_17 < 4; ++ax3_outer_17) {
        for (int32_t ax3_inner_17 = 0; ax3_inner_17 < 16; ++ax3_inner_17) {
          T_matmul_NN[((((ax0_ax1_fused_20 * 512) + (ax2_17 * 64)) + (ax3_outer_17 * 16)) + ax3_inner_17)] = T_multiply[((((ax2_17 * 8192) + (ax0_ax1_fused_20 * 64)) + (ax3_outer_17 * 16)) + ax3_inner_17)];
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_3 = 0; i0_3 < 128; ++i0_3) {
    for (int32_t i1_outer_3 = 0; i1_outer_3 < 32; ++i1_outer_3) {
      for (int32_t i1_inner_3 = 0; i1_inner_3 < 16; ++i1_inner_3) {
        T_transpose[(((i0_3 * 512) + (i1_outer_3 * 16)) + i1_inner_3)] = 0.000000e+00f;
        for (int32_t k_6 = 0; k_6 < 512; ++k_6) {
          T_transpose[(((i0_3 * 512) + (i1_outer_3 * 16)) + i1_inner_3)] = (T_transpose[(((i0_3 * 512) + (i1_outer_3 * 16)) + i1_inner_3)] + (T_matmul_NN[((i0_3 * 512) + k_6)] * ph_7[(((k_6 * 512) + (i1_outer_3 * 16)) + i1_inner_3)]));
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_21 = 0; ax0_ax1_fused_21 < 128; ++ax0_ax1_fused_21) {
    for (int32_t ax2_outer_3 = 0; ax2_outer_3 < 32; ++ax2_outer_3) {
      for (int32_t ax2_inner_3 = 0; ax2_inner_3 < 16; ++ax2_inner_3) {
        T_transpose[(((ax0_ax1_fused_21 * 512) + (ax2_outer_3 * 16)) + ax2_inner_3)] = (T_transpose[(((ax0_ax1_fused_21 * 512) + (ax2_outer_3 * 16)) + ax2_inner_3)] + T_cast[(((ax0_ax1_fused_21 * 512) + (ax2_outer_3 * 16)) + ax2_inner_3)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_outer_1 = 0; ax0_outer_1 < 32; ++ax0_outer_1) {
    for (int32_t ax0_inner_1 = 0; ax0_inner_1 < 16; ++ax0_inner_1) {
      T_cast_1[((ax0_outer_1 * 16) + ax0_inner_1)] = ph_8[((ax0_outer_1 * 16) + ax0_inner_1)];
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_22 = 0; ax0_ax1_fused_22 < 128; ++ax0_ax1_fused_22) {
    for (int32_t ax2_outer_4 = 0; ax2_outer_4 < 32; ++ax2_outer_4) {
      for (int32_t ax2_inner_4 = 0; ax2_inner_4 < 16; ++ax2_inner_4) {
        T_multiply[(((ax0_ax1_fused_22 * 512) + (ax2_outer_4 * 16)) + ax2_inner_4)] = (T_transpose[(((ax0_ax1_fused_22 * 512) + (ax2_outer_4 * 16)) + ax2_inner_4)] * T_transpose[(((ax0_ax1_fused_22 * 512) + (ax2_outer_4 * 16)) + ax2_inner_4)]);
      }
    }
  }
  for (int32_t ax1_outer_2 = 0; ax1_outer_2 < 8; ++ax1_outer_2) {
    for (int32_t ax1_inner_2 = 0; ax1_inner_2 < 16; ++ax1_inner_2) {
      T_multiply_red[((ax1_outer_2 * 16) + ax1_inner_2)] = 0.000000e+00f;
      for (int32_t k2_1 = 0; k2_1 < 512; ++k2_1) {
        T_multiply_red[((ax1_outer_2 * 16) + ax1_inner_2)] = (T_multiply_red[((ax1_outer_2 * 16) + ax1_inner_2)] + T_multiply[(((ax1_outer_2 * 8192) + (ax1_inner_2 * 512)) + k2_1)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_23 = 0; ax0_ax1_fused_23 < 128; ++ax0_ax1_fused_23) {
    for (int32_t ax2_outer_5 = 0; ax2_outer_5 < 32; ++ax2_outer_5) {
      for (int32_t ax2_inner_5 = 0; ax2_inner_5 < 16; ++ax2_inner_5) {
        T_transpose[(((ax0_ax1_fused_23 * 512) + (ax2_outer_5 * 16)) + ax2_inner_5)] = ((T_transpose[(((ax0_ax1_fused_23 * 512) + (ax2_outer_5 * 16)) + ax2_inner_5)] * T_cast_1[((ax2_outer_5 * 16) + ax2_inner_5)]) * (1.000000e+00f / sqrtf(((T_multiply_red[ax0_ax1_fused_23] * 1.953125e-03f) + 1.000000e-05f))));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_1 = 0; ax0_1 < 48; ++ax0_1) {
    for (int32_t ax1_outer_3 = 0; ax1_outer_3 < 32; ++ax1_outer_3) {
      for (int32_t ax1_inner_3 = 0; ax1_inner_3 < 16; ++ax1_inner_3) {
        T_strided_slice[(((ax0_1 * 512) + (ax1_outer_3 * 16)) + ax1_inner_3)] = T_transpose[(((ax0_1 * 512) + (ax1_outer_3 * 16)) + ax1_inner_3)];
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_4 = 0; i0_4 < 48; ++i0_4) {
    for (int32_t i1_outer_4 = 0; i1_outer_4 < 86; ++i1_outer_4) {
      for (int32_t i1_inner_4 = 0; i1_inner_4 < 16; ++i1_inner_4) {
        T_strided_slice_1[(((i0_4 * 1376) + (i1_outer_4 * 16)) + i1_inner_4)] = 0.000000e+00f;
        for (int32_t k_7 = 0; k_7 < 512; ++k_7) {
          T_strided_slice_1[(((i0_4 * 1376) + (i1_outer_4 * 16)) + i1_inner_4)] = (T_strided_slice_1[(((i0_4 * 1376) + (i1_outer_4 * 16)) + i1_inner_4)] + (T_strided_slice[((i0_4 * 512) + k_7)] * ph_9[(((k_7 * 1376) + (i1_outer_4 * 16)) + i1_inner_4)]));
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_5 = 0; i0_5 < 48; ++i0_5) {
    for (int32_t i1_outer_5 = 0; i1_outer_5 < 86; ++i1_outer_5) {
      for (int32_t i1_inner_5 = 0; i1_inner_5 < 16; ++i1_inner_5) {
        T_multiply[(((i0_5 * 1376) + (i1_outer_5 * 16)) + i1_inner_5)] = (1.000000e+00f / (1.000000e+00f + expf((0.000000e+00f - T_strided_slice_1[(((i0_5 * 1376) + (i1_outer_5 * 16)) + i1_inner_5)]))));
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_2 = 0; ax0_2 < 48; ++ax0_2) {
    for (int32_t ax1_outer_4 = 0; ax1_outer_4 < 86; ++ax1_outer_4) {
      for (int32_t ax1_inner_4 = 0; ax1_inner_4 < 16; ++ax1_inner_4) {
        T_multiply[(((ax0_2 * 1376) + (ax1_outer_4 * 16)) + ax1_inner_4)] = (T_strided_slice_1[(((ax0_2 * 1376) + (ax1_outer_4 * 16)) + ax1_inner_4)] * T_multiply[(((ax0_2 * 1376) + (ax1_outer_4 * 16)) + ax1_inner_4)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_6 = 0; i0_6 < 48; ++i0_6) {
    for (int32_t i1_outer_6 = 0; i1_outer_6 < 86; ++i1_outer_6) {
      for (int32_t i1_inner_6 = 0; i1_inner_6 < 16; ++i1_inner_6) {
        T_strided_slice_1[(((i0_6 * 1376) + (i1_outer_6 * 16)) + i1_inner_6)] = 0.000000e+00f;
        for (int32_t k_8 = 0; k_8 < 512; ++k_8) {
          T_strided_slice_1[(((i0_6 * 1376) + (i1_outer_6 * 16)) + i1_inner_6)] = (T_strided_slice_1[(((i0_6 * 1376) + (i1_outer_6 * 16)) + i1_inner_6)] + (T_strided_slice[((i0_6 * 512) + k_8)] * ph_10[(((k_8 * 1376) + (i1_outer_6 * 16)) + i1_inner_6)]));
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_3 = 0; ax0_3 < 48; ++ax0_3) {
    for (int32_t ax1_outer_5 = 0; ax1_outer_5 < 86; ++ax1_outer_5) {
      for (int32_t ax1_inner_5 = 0; ax1_inner_5 < 16; ++ax1_inner_5) {
        T_multiply[(((ax0_3 * 1376) + (ax1_outer_5 * 16)) + ax1_inner_5)] = (T_multiply[(((ax0_3 * 1376) + (ax1_outer_5 * 16)) + ax1_inner_5)] * T_strided_slice_1[(((ax0_3 * 1376) + (ax1_outer_5 * 16)) + ax1_inner_5)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_7 = 0; i0_7 < 48; ++i0_7) {
    for (int32_t i1_outer_7 = 0; i1_outer_7 < 86; ++i1_outer_7) {
      for (int32_t i1_inner_7 = 0; i1_inner_7 < 16; ++i1_inner_7) {
        T_strided_slice_1[(((i0_7 * 1376) + (i1_outer_7 * 16)) + i1_inner_7)] = 0.000000e+00f;
        for (int32_t k_9 = 0; k_9 < 1376; ++k_9) {
          T_strided_slice_1[(((i0_7 * 1376) + (i1_outer_7 * 16)) + i1_inner_7)] = (T_strided_slice_1[(((i0_7 * 1376) + (i1_outer_7 * 16)) + i1_inner_7)] + (T_multiply[((i0_7 * 1376) + k_9)] * ph_11[(((k_9 * 1376) + (i1_outer_7 * 16)) + i1_inner_7)]));
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_24 = 0; ax0_ax1_fused_24 < 128; ++ax0_ax1_fused_24) {
    for (int32_t ax2_outer_6 = 0; ax2_outer_6 < 32; ++ax2_outer_6) {
      for (int32_t ax2_inner_6 = 0; ax2_inner_6 < 16; ++ax2_inner_6) {
        T_matmul_NN[(((ax0_ax1_fused_24 * 512) + (ax2_outer_6 * 16)) + ax2_inner_6)] = T_strided_slice_1[(((ax0_ax1_fused_24 * 512) + (ax2_outer_6 * 16)) + ax2_inner_6)];
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused_25 = 0; ax0_ax1_fused_25 < 128; ++ax0_ax1_fused_25) {
    for (int32_t ax2_outer_7 = 0; ax2_outer_7 < 32; ++ax2_outer_7) {
      for (int32_t ax2_inner_7 = 0; ax2_inner_7 < 16; ++ax2_inner_7) {
        T_add[(((ax0_ax1_fused_25 * 512) + (ax2_outer_7 * 16)) + ax2_inner_7)] = (T_transpose[(((ax0_ax1_fused_25 * 512) + (ax2_outer_7 * 16)) + ax2_inner_7)] + T_matmul_NN[(((ax0_ax1_fused_25 * 512) + (ax2_outer_7 * 16)) + ax2_inner_7)]);
      }
    }
  }
}

