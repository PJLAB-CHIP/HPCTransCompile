void default_function_kernel(float* T_fast_exp, float* ph) {
  float T_softmax_maxelem[1265];
  float compute[1265];
  for (int32_t i0 = 0; i0 < 11; ++i0) {
    for (int32_t i1 = 0; i1 < 23; ++i1) {
      for (int32_t i2 = 0; i2 < 5; ++i2) {
        T_softmax_maxelem[(((i0 * 115) + (i1 * 5)) + i2)] = -3.402823e+38f;
        for (int32_t k = 0; k < 12; ++k) {
          T_softmax_maxelem[(((i0 * 115) + (i1 * 5)) + i2)] = max(T_softmax_maxelem[(((i0 * 115) + (i1 * 5)) + i2)], ph[((((i1 * 60) + k) + 48) - (i2 * 12))]);
        }
      }
    }
  }
  for (int32_t i0_1 = 0; i0_1 < 11; ++i0_1) {
    for (int32_t i1_1 = 0; i1_1 < 23; ++i1_1) {
      for (int32_t i2_1 = 0; i2_1 < 5; ++i2_1) {
        compute[(((i0_1 * 115) + (i1_1 * 5)) + i2_1)] = 0.000000e+00f;
        for (int32_t k_1 = 0; k_1 < 12; ++k_1) {
          compute[(((i0_1 * 115) + (i1_1 * 5)) + i2_1)] = (compute[(((i0_1 * 115) + (i1_1 * 5)) + i2_1)] + expf((ph[((((i1_1 * 60) + k_1) + 48) - (i2_1 * 12))] - T_softmax_maxelem[(((i0_1 * 115) + (i1_1 * 5)) + i2_1)])));
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 253; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 5; ++ax2) {
      for (int32_t ax3_inner = 0; ax3_inner < 12; ++ax3_inner) {
          int32_t v_ = ((int32_t)(floorf(((max(min(((ph[(((((ax0_ax1_fused % 23) * 60) + ax3_inner) + 48) - (ax2 * 12))] - T_softmax_maxelem[((ax0_ax1_fused * 5) + ax2)]) - logf(compute[((ax0_ax1_fused * 5) + ax2)])), 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) + 1.270000e+02f)) << 23;
        T_fast_exp[(((ax0_ax1_fused * 60) + (ax2 * 12)) + ax3_inner)] = max(((*(float *)(&(v_))) * ((((((((((((((1.987569e-04f * (max(min(((ph[(((((ax0_ax1_fused % 23) * 60) + ax3_inner) + 48) - (ax2 * 12))] - T_softmax_maxelem[((ax0_ax1_fused * 5) + ax2)]) - logf(compute[((ax0_ax1_fused * 5) + ax2)])), 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(((ph[(((((ax0_ax1_fused % 23) * 60) + ax3_inner) + 48) - (ax2 * 12))] - T_softmax_maxelem[((ax0_ax1_fused * 5) + ax2)]) - logf(compute[((ax0_ax1_fused * 5) + ax2)])), 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.398200e-03f) * (max(min(((ph[(((((ax0_ax1_fused % 23) * 60) + ax3_inner) + 48) - (ax2 * 12))] - T_softmax_maxelem[((ax0_ax1_fused * 5) + ax2)]) - logf(compute[((ax0_ax1_fused * 5) + ax2)])), 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(((ph[(((((ax0_ax1_fused % 23) * 60) + ax3_inner) + 48) - (ax2 * 12))] - T_softmax_maxelem[((ax0_ax1_fused * 5) + ax2)]) - logf(compute[((ax0_ax1_fused * 5) + ax2)])), 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 8.333452e-03f) * (max(min(((ph[(((((ax0_ax1_fused % 23) * 60) + ax3_inner) + 48) - (ax2 * 12))] - T_softmax_maxelem[((ax0_ax1_fused * 5) + ax2)]) - logf(compute[((ax0_ax1_fused * 5) + ax2)])), 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(((ph[(((((ax0_ax1_fused % 23) * 60) + ax3_inner) + 48) - (ax2 * 12))] - T_softmax_maxelem[((ax0_ax1_fused * 5) + ax2)]) - logf(compute[((ax0_ax1_fused * 5) + ax2)])), 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 4.166580e-02f) * (max(min(((ph[(((((ax0_ax1_fused % 23) * 60) + ax3_inner) + 48) - (ax2 * 12))] - T_softmax_maxelem[((ax0_ax1_fused * 5) + ax2)]) - logf(compute[((ax0_ax1_fused * 5) + ax2)])), 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(((ph[(((((ax0_ax1_fused % 23) * 60) + ax3_inner) + 48) - (ax2 * 12))] - T_softmax_maxelem[((ax0_ax1_fused * 5) + ax2)]) - logf(compute[((ax0_ax1_fused * 5) + ax2)])), 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.666667e-01f) * (max(min(((ph[(((((ax0_ax1_fused % 23) * 60) + ax3_inner) + 48) - (ax2 * 12))] - T_softmax_maxelem[((ax0_ax1_fused * 5) + ax2)]) - logf(compute[((ax0_ax1_fused * 5) + ax2)])), 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(((ph[(((((ax0_ax1_fused % 23) * 60) + ax3_inner) + 48) - (ax2 * 12))] - T_softmax_maxelem[((ax0_ax1_fused * 5) + ax2)]) - logf(compute[((ax0_ax1_fused * 5) + ax2)])), 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 5.000000e-01f) * (max(min(((ph[(((((ax0_ax1_fused % 23) * 60) + ax3_inner) + 48) - (ax2 * 12))] - T_softmax_maxelem[((ax0_ax1_fused * 5) + ax2)]) - logf(compute[((ax0_ax1_fused * 5) + ax2)])), 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(((ph[(((((ax0_ax1_fused % 23) * 60) + ax3_inner) + 48) - (ax2 * 12))] - T_softmax_maxelem[((ax0_ax1_fused * 5) + ax2)]) - logf(compute[((ax0_ax1_fused * 5) + ax2)])), 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) * (max(min(((ph[(((((ax0_ax1_fused % 23) * 60) + ax3_inner) + 48) - (ax2 * 12))] - T_softmax_maxelem[((ax0_ax1_fused * 5) + ax2)]) - logf(compute[((ax0_ax1_fused * 5) + ax2)])), 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(((ph[(((((ax0_ax1_fused % 23) * 60) + ax3_inner) + 48) - (ax2 * 12))] - T_softmax_maxelem[((ax0_ax1_fused * 5) + ax2)]) - logf(compute[((ax0_ax1_fused * 5) + ax2)])), 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + (max(min(((ph[(((((ax0_ax1_fused % 23) * 60) + ax3_inner) + 48) - (ax2 * 12))] - T_softmax_maxelem[((ax0_ax1_fused * 5) + ax2)]) - logf(compute[((ax0_ax1_fused * 5) + ax2)])), 8.837627e+01f), -8.837626e+01f) - (floorf(((max(min(((ph[(((((ax0_ax1_fused % 23) * 60) + ax3_inner) + 48) - (ax2 * 12))] - T_softmax_maxelem[((ax0_ax1_fused * 5) + ax2)]) - logf(compute[((ax0_ax1_fused * 5) + ax2)])), 8.837627e+01f), -8.837626e+01f) * 1.442695e+00f) + 5.000000e-01f)) * 6.931472e-01f))) + 1.000000e+00f)), ((ph[(((((ax0_ax1_fused % 23) * 60) + ax3_inner) + 48) - (ax2 * 12))] - T_softmax_maxelem[((ax0_ax1_fused * 5) + ax2)]) - logf(compute[((ax0_ax1_fused * 5) + ax2)])));
      }
    }
  }
}

