void default_function_kernel(float* T_add, float* compute, float* ph_0, float* ph_3) {
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 20; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 8; ++ax1) {
      for (int32_t ax2 = 0; ax2 < 16; ++ax2) {
        T_add[(((ax0 * 128) + (ax1 * 16)) + ax2)] = (ph_0[(((ax0 * 128) + (ax1 * 16)) + ax2)] + ph_3[(((ax0 * 128) + (ax1 * 16)) + ax2)]);
      }
    }
  }
  #pragma omp parallel for
  for (int32_t i0_i1_fused = 0; i0_i1_fused < 160; ++i0_i1_fused) {
    for (int32_t i2 = 0; i2 < 16; ++i2) {
      compute[((i0_i1_fused * 16) + i2)] = asinhf((ph_0[((i0_i1_fused * 16) + i2)] / (ph_0[((i0_i1_fused * 16) + i2)] - ph_3[((i0_i1_fused * 16) + i2)])));
    }
  }
}

