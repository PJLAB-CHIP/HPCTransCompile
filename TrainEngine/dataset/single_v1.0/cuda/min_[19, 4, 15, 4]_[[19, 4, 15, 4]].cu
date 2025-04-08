extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ data, float* __restrict__ data_red) {
  float normal_reduce_temp0[1];
  float red_buf0[1];
  normal_reduce_temp0[0] = 3.402823e+38f;
  for (int k0_k1_fused_k2_fused_k3_fused_outer = 0; k0_k1_fused_k2_fused_k3_fused_outer < 143; ++k0_k1_fused_k2_fused_k3_fused_outer) {
    if (((k0_k1_fused_k2_fused_k3_fused_outer * 2) + (((int)threadIdx.x) >> 4)) < 285) {
      normal_reduce_temp0[0] = min(normal_reduce_temp0[0], data[((k0_k1_fused_k2_fused_k3_fused_outer * 32) + ((int)threadIdx.x))]);
    }
  }
  uint mask[1];
  float t0[1];
  red_buf0[0] = normal_reduce_temp0[0];
  mask[0] = __activemask();
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 16, 32);
  red_buf0[0] = min(red_buf0[0], t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
  red_buf0[0] = min(red_buf0[0], t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  red_buf0[0] = min(red_buf0[0], t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = min(red_buf0[0], t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = min(red_buf0[0], t0[0]);
  red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], 0, 32);
  if (((int)threadIdx.x) == 0) {
    data_red[0] = red_buf0[0];
  }
}

