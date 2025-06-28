extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = asinhf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(18) default_function_kernel_3(float* __restrict__ compute, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  float T_batch_matmul_NN[10];
  __shared__ float ph_3_shared[180];
  T_batch_matmul_NN[0] = 0.000000e+00f;
  T_batch_matmul_NN[1] = 0.000000e+00f;
  T_batch_matmul_NN[2] = 0.000000e+00f;
  T_batch_matmul_NN[3] = 0.000000e+00f;
  T_batch_matmul_NN[4] = 0.000000e+00f;
  T_batch_matmul_NN[5] = 0.000000e+00f;
  T_batch_matmul_NN[6] = 0.000000e+00f;
  T_batch_matmul_NN[7] = 0.000000e+00f;
  T_batch_matmul_NN[8] = 0.000000e+00f;
  T_batch_matmul_NN[9] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 2; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 5; ++ax0_ax1_fused_ax2_fused_outer_outer) {
      *(float2*)(ph_3_shared + ((ax0_ax1_fused_ax2_fused_outer_outer * 36) + (((int)threadIdx.x) * 2))) = *(float2*)(ph_3 + ((((((ax0_ax1_fused_ax2_fused_outer_outer * 36) + (((int)threadIdx.x) * 2)) / 20) * 40) + (k_outer_outer * 20)) + (((ax0_ax1_fused_ax2_fused_outer_outer * 16) + (((int)threadIdx.x) * 2)) % 20)));
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
      for (int k_inner = 0; k_inner < 2; ++k_inner) {
        T_batch_matmul_NN[0] = (T_batch_matmul_NN[0] + (ph_0[(((((((((int)threadIdx.x) >> 1) * 64) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 1) * 8)) + (k_outer_outer * 4)) + (k_outer_inner * 2)) + k_inner)] * ph_3_shared[((((((int)threadIdx.x) >> 1) * 20) + (k_outer_inner * 10)) + (k_inner * 5))]));
        T_batch_matmul_NN[1] = (T_batch_matmul_NN[1] + (ph_0[(((((((((int)threadIdx.x) >> 1) * 64) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 1) * 8)) + (k_outer_outer * 4)) + (k_outer_inner * 2)) + k_inner)] * ph_3_shared[(((((((int)threadIdx.x) >> 1) * 20) + (k_outer_inner * 10)) + (k_inner * 5)) + 1)]));
        T_batch_matmul_NN[2] = (T_batch_matmul_NN[2] + (ph_0[(((((((((int)threadIdx.x) >> 1) * 64) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 1) * 8)) + (k_outer_outer * 4)) + (k_outer_inner * 2)) + k_inner)] * ph_3_shared[(((((((int)threadIdx.x) >> 1) * 20) + (k_outer_inner * 10)) + (k_inner * 5)) + 2)]));
        T_batch_matmul_NN[3] = (T_batch_matmul_NN[3] + (ph_0[(((((((((int)threadIdx.x) >> 1) * 64) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 1) * 8)) + (k_outer_outer * 4)) + (k_outer_inner * 2)) + k_inner)] * ph_3_shared[(((((((int)threadIdx.x) >> 1) * 20) + (k_outer_inner * 10)) + (k_inner * 5)) + 3)]));
        T_batch_matmul_NN[4] = (T_batch_matmul_NN[4] + (ph_0[(((((((((int)threadIdx.x) >> 1) * 64) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 1) * 8)) + (k_outer_outer * 4)) + (k_outer_inner * 2)) + k_inner)] * ph_3_shared[(((((((int)threadIdx.x) >> 1) * 20) + (k_outer_inner * 10)) + (k_inner * 5)) + 4)]));
        T_batch_matmul_NN[5] = (T_batch_matmul_NN[5] + (ph_0[((((((((((int)threadIdx.x) >> 1) * 64) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 1) * 8)) + (k_outer_outer * 4)) + (k_outer_inner * 2)) + k_inner) + 16)] * ph_3_shared[((((((int)threadIdx.x) >> 1) * 20) + (k_outer_inner * 10)) + (k_inner * 5))]));
        T_batch_matmul_NN[6] = (T_batch_matmul_NN[6] + (ph_0[((((((((((int)threadIdx.x) >> 1) * 64) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 1) * 8)) + (k_outer_outer * 4)) + (k_outer_inner * 2)) + k_inner) + 16)] * ph_3_shared[(((((((int)threadIdx.x) >> 1) * 20) + (k_outer_inner * 10)) + (k_inner * 5)) + 1)]));
        T_batch_matmul_NN[7] = (T_batch_matmul_NN[7] + (ph_0[((((((((((int)threadIdx.x) >> 1) * 64) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 1) * 8)) + (k_outer_outer * 4)) + (k_outer_inner * 2)) + k_inner) + 16)] * ph_3_shared[(((((((int)threadIdx.x) >> 1) * 20) + (k_outer_inner * 10)) + (k_inner * 5)) + 2)]));
        T_batch_matmul_NN[8] = (T_batch_matmul_NN[8] + (ph_0[((((((((((int)threadIdx.x) >> 1) * 64) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 1) * 8)) + (k_outer_outer * 4)) + (k_outer_inner * 2)) + k_inner) + 16)] * ph_3_shared[(((((((int)threadIdx.x) >> 1) * 20) + (k_outer_inner * 10)) + (k_inner * 5)) + 3)]));
        T_batch_matmul_NN[9] = (T_batch_matmul_NN[9] + (ph_0[((((((((((int)threadIdx.x) >> 1) * 64) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 1) * 8)) + (k_outer_outer * 4)) + (k_outer_inner * 2)) + k_inner) + 16)] * ph_3_shared[(((((((int)threadIdx.x) >> 1) * 20) + (k_outer_inner * 10)) + (k_inner * 5)) + 4)]));
      }
    }
  }
  compute[((((((int)threadIdx.x) >> 1) * 40) + (((int)blockIdx.x) * 20)) + ((((int)threadIdx.x) & 1) * 5))] = __sinf(T_batch_matmul_NN[0]);
  compute[(((((((int)threadIdx.x) >> 1) * 40) + (((int)blockIdx.x) * 20)) + ((((int)threadIdx.x) & 1) * 5)) + 1)] = __sinf(T_batch_matmul_NN[1]);
  compute[(((((((int)threadIdx.x) >> 1) * 40) + (((int)blockIdx.x) * 20)) + ((((int)threadIdx.x) & 1) * 5)) + 2)] = __sinf(T_batch_matmul_NN[2]);
  compute[(((((((int)threadIdx.x) >> 1) * 40) + (((int)blockIdx.x) * 20)) + ((((int)threadIdx.x) & 1) * 5)) + 3)] = __sinf(T_batch_matmul_NN[3]);
  compute[(((((((int)threadIdx.x) >> 1) * 40) + (((int)blockIdx.x) * 20)) + ((((int)threadIdx.x) & 1) * 5)) + 4)] = __sinf(T_batch_matmul_NN[4]);
  compute[(((((((int)threadIdx.x) >> 1) * 40) + (((int)blockIdx.x) * 20)) + ((((int)threadIdx.x) & 1) * 5)) + 10)] = __sinf(T_batch_matmul_NN[5]);
  compute[(((((((int)threadIdx.x) >> 1) * 40) + (((int)blockIdx.x) * 20)) + ((((int)threadIdx.x) & 1) * 5)) + 11)] = __sinf(T_batch_matmul_NN[6]);
  compute[(((((((int)threadIdx.x) >> 1) * 40) + (((int)blockIdx.x) * 20)) + ((((int)threadIdx.x) & 1) * 5)) + 12)] = __sinf(T_batch_matmul_NN[7]);
  compute[(((((((int)threadIdx.x) >> 1) * 40) + (((int)blockIdx.x) * 20)) + ((((int)threadIdx.x) & 1) * 5)) + 13)] = __sinf(T_batch_matmul_NN[8]);
  compute[(((((((int)threadIdx.x) >> 1) * 40) + (((int)blockIdx.x) * 20)) + ((((int)threadIdx.x) & 1) * 5)) + 14)] = __sinf(T_batch_matmul_NN[9]);
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_1(float* __restrict__ T_divide, float* __restrict__ ph_0) {
  T_divide[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (fabsf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]) / ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_2(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = ceilf(fabsf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]));
}

