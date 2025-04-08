void default_function_kernel(float* T_multiply, float* compute, float* ph_0) {
  #pragma omp parallel for
  for (int32_t i0 = 0; i0 < 6; ++i0) {
    float compute_1[1];
    for (int32_t i1 = 0; i1 < 7; ++i1) {
      compute_1[0] = expf(ph_0[((i0 * 7) + i1)]);
      compute[((i0 * 7) + i1)] = cosf((ph_0[((i0 * 7) + i1)] + compute_1[0]));
    }
  }
  #pragma omp parallel for
  for (int32_t ax0 = 0; ax0 < 6; ++ax0) {
    for (int32_t ax1 = 0; ax1 < 7; ++ax1) {
      T_multiply[((ax0 * 7) + ax1)] = (ph_0[((ax0 * 7) + ax1)] * atanhf(ph_0[((ax0 * 7) + ax1)]));
    }
  }
}

