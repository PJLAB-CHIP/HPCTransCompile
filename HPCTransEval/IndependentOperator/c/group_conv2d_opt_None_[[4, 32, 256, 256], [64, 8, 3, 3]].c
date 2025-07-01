void default_function_kernel(float* A, float* W, float* output_unpack) {
  float data_vec[8388608];
  float kernel_vec[4608];
  float conv_global[16516096];
  for (int32_t g = 0; g < 4; ++g) {
    #pragma omp parallel for
    for (int32_t n_C_fused_h_fused = 0; n_C_fused_h_fused < 2048; ++n_C_fused_h_fused) {
      for (int32_t c = 0; c < 4; ++c) {
        for (int32_t w = 0; w < 256; ++w) {
          data_vec[((((g * 2097152) + (n_C_fused_h_fused * 1024)) + (c * 256)) + w)] = A[(((((((n_C_fused_h_fused >> 9) * 2097152) + (g * 524288)) + (((n_C_fused_h_fused & 511) >> 8) * 262144)) + (c * 65536)) + ((n_C_fused_h_fused & 255) * 256)) + w)];
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t g_out_channel_fused_h_fused = 0; g_out_channel_fused_h_fused < 48; ++g_out_channel_fused_h_fused) {
    for (int32_t in_channel = 0; in_channel < 2; ++in_channel) {
      for (int32_t w_1 = 0; w_1 < 3; ++w_1) {
        for (int32_t ci = 0; ci < 4; ++ci) {
          for (int32_t co = 0; co < 4; ++co) {
            kernel_vec[(((((((g_out_channel_fused_h_fused / 3) * 288) + (in_channel * 144)) + ((g_out_channel_fused_h_fused % 3) * 48)) + (w_1 * 16)) + (ci * 4)) + co)] = W[(((((((g_out_channel_fused_h_fused / 3) * 288) + (co * 72)) + (in_channel * 36)) + (ci * 9)) + ((g_out_channel_fused_h_fused % 3) * 3)) + w_1)];
          }
        }
      }
    }
  }
  #pragma omp parallel for
  for (int32_t g_c_n_c_fused_oc_chunk_c_fused_oh_c_fused = 0; g_c_n_c_fused_oc_chunk_c_fused_oh_c_fused < 16256; ++g_c_n_c_fused_oc_chunk_c_fused_oh_c_fused) {
    for (int32_t ow_c_outer = 0; ow_c_outer < 127; ++ow_c_outer) {
      for (int32_t oc_block_c_init = 0; oc_block_c_init < 4; ++oc_block_c_init) {
        conv_global[(((g_c_n_c_fused_oc_chunk_c_fused_oh_c_fused * 1016) + (ow_c_outer * 8)) + oc_block_c_init)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init_1 = 0; oc_block_c_init_1 < 4; ++oc_block_c_init_1) {
        conv_global[((((g_c_n_c_fused_oc_chunk_c_fused_oh_c_fused * 1016) + (ow_c_outer * 8)) + oc_block_c_init_1) + 4)] = 0.000000e+00f;
      }
      for (int32_t ic_outer = 0; ic_outer < 2; ++ic_outer) {
        for (int32_t kh = 0; kh < 3; ++kh) {
          for (int32_t kw = 0; kw < 3; ++kw) {
            for (int32_t ic_inner = 0; ic_inner < 4; ++ic_inner) {
              for (int32_t oc_block_c = 0; oc_block_c < 4; ++oc_block_c) {
                conv_global[(((g_c_n_c_fused_oc_chunk_c_fused_oh_c_fused * 1016) + (ow_c_outer * 8)) + oc_block_c)] = (conv_global[(((g_c_n_c_fused_oc_chunk_c_fused_oh_c_fused * 1016) + (ow_c_outer * 8)) + oc_block_c)] + (data_vec[((((((((g_c_n_c_fused_oc_chunk_c_fused_oh_c_fused / 1016) * 524288) + (ic_outer * 262144)) + (kh * 1024)) + ((g_c_n_c_fused_oc_chunk_c_fused_oh_c_fused % 254) * 1024)) + (ic_inner * 256)) + (ow_c_outer * 2)) + kw)] * kernel_vec[((((((((g_c_n_c_fused_oc_chunk_c_fused_oh_c_fused / 4064) * 1152) + (((g_c_n_c_fused_oc_chunk_c_fused_oh_c_fused % 1016) / 254) * 288)) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c)]));
              }
              for (int32_t oc_block_c_1 = 0; oc_block_c_1 < 4; ++oc_block_c_1) {
                conv_global[((((g_c_n_c_fused_oc_chunk_c_fused_oh_c_fused * 1016) + (ow_c_outer * 8)) + oc_block_c_1) + 4)] = (conv_global[((((g_c_n_c_fused_oc_chunk_c_fused_oh_c_fused * 1016) + (ow_c_outer * 8)) + oc_block_c_1) + 4)] + (data_vec[(((((((((g_c_n_c_fused_oc_chunk_c_fused_oh_c_fused / 1016) * 524288) + (ic_outer * 262144)) + (kh * 1024)) + ((g_c_n_c_fused_oc_chunk_c_fused_oh_c_fused % 254) * 1024)) + (ic_inner * 256)) + (ow_c_outer * 2)) + kw) + 1)] * kernel_vec[((((((((g_c_n_c_fused_oc_chunk_c_fused_oh_c_fused / 4064) * 1152) + (((g_c_n_c_fused_oc_chunk_c_fused_oh_c_fused % 1016) / 254) * 288)) + (ic_outer * 144)) + (kh * 48)) + (kw * 16)) + (ic_inner * 4)) + oc_block_c_1)]));
              }
            }
          }
        }
      }
    }
  }
  for (int32_t n = 0; n < 4; ++n) {
    #pragma omp parallel for
    for (int32_t c_outer_h_fused = 0; c_outer_h_fused < 4064; ++c_outer_h_fused) {
      for (int32_t w_outer = 0; w_outer < 127; ++w_outer) {
        for (int32_t w_inner = 0; w_inner < 2; ++w_inner) {
          for (int32_t c_inner = 0; c_inner < 4; ++c_inner) {
            output_unpack[((((((n * 4129024) + ((c_outer_h_fused / 254) * 258064)) + (c_inner * 64516)) + ((c_outer_h_fused % 254) * 254)) + (w_outer * 2)) + w_inner)] = conv_global[(((((((c_outer_h_fused / 1016) * 4129024) + (n * 1032256)) + ((c_outer_h_fused % 1016) * 1016)) + (w_outer * 8)) + (w_inner * 4)) + c_inner)];
          }
        }
      }
    }
  }
}

