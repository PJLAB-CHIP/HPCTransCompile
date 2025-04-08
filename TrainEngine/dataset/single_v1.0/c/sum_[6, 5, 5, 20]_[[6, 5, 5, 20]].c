void default_function_kernel(float* data, float* data_red) {
  float data_red_rf[15];
  for (int32_t k0_k1_fused_k2_fused_k3_fused_inner_init = 0; k0_k1_fused_k2_fused_k3_fused_inner_init < 15; ++k0_k1_fused_k2_fused_k3_fused_inner_init) {
    data_red_rf[k0_k1_fused_k2_fused_k3_fused_inner_init] = 0.000000e+00f;
  }
  for (int32_t k0_k1_fused_k2_fused_k3_fused_outer = 0; k0_k1_fused_k2_fused_k3_fused_outer < 200; ++k0_k1_fused_k2_fused_k3_fused_outer) {
    for (int32_t k0_k1_fused_k2_fused_k3_fused_inner = 0; k0_k1_fused_k2_fused_k3_fused_inner < 15; ++k0_k1_fused_k2_fused_k3_fused_inner) {
      data_red_rf[k0_k1_fused_k2_fused_k3_fused_inner] = (data_red_rf[k0_k1_fused_k2_fused_k3_fused_inner] + data[((k0_k1_fused_k2_fused_k3_fused_outer * 15) + k0_k1_fused_k2_fused_k3_fused_inner)]);
    }
  }
  data_red[0] = 0.000000e+00f;
  for (int32_t k0_k1_fused_k2_fused_k3_fused_inner_v = 0; k0_k1_fused_k2_fused_k3_fused_inner_v < 15; ++k0_k1_fused_k2_fused_k3_fused_inner_v) {
    data_red[0] = (data_red[0] + data_red_rf[k0_k1_fused_k2_fused_k3_fused_inner_v]);
  }
}

