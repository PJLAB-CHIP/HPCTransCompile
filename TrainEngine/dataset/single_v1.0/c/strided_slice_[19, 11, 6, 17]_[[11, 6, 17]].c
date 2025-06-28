void default_function_kernel(float* T_strided_slice, float* a) {
  #pragma omp parallel for
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 12; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 7; ++ax2) {
      T_strided_slice[((ax0_ax1_fused * 7) + ax2)] = a[(((((ax0_ax1_fused >> 2) * 102) + ((ax0_ax1_fused & 3) * 17)) + ax2) + 139)];
    }
  }
}

