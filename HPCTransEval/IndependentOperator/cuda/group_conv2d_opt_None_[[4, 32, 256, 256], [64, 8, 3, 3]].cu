extern "C" __global__ void default_function_kernel(float* __restrict__ A, float* __restrict__ W, float* __restrict__ group_conv2d_nchw) {
  float group_conv2d_nchw_local[1];
  __shared__ float pad_temp_shared[1];
  __shared__ float W_shared[1];
  group_conv2d_nchw_local[0] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
    for (int ry_outer = 0; ry_outer < 3; ++ry_outer) {
      for (int rx_outer = 0; rx_outer < 3; ++rx_outer) {
        __syncthreads();
        pad_temp_shared[0] = A[(((((((((int)blockIdx.z) * 2097152) + ((((int)blockIdx.y) >> 4) * 524288)) + (rc_outer * 65536)) + ((((int)blockIdx.x) / 254) * 256)) + (ry_outer * 256)) + rx_outer) + (((int)blockIdx.x) % 254))];
        W_shared[0] = W[((((((int)blockIdx.y) * 72) + (rc_outer * 9)) + (ry_outer * 3)) + rx_outer)];
        __syncthreads();
        group_conv2d_nchw_local[0] = (group_conv2d_nchw_local[0] + (pad_temp_shared[0] * W_shared[0]));
      }
    }
  }
  group_conv2d_nchw[(((((int)blockIdx.z) * 4129024) + (((int)blockIdx.y) * 64516)) + ((int)blockIdx.x))] = group_conv2d_nchw_local[0];
}

