extern "C" __global__ void __launch_bounds__(32) default_function_kernel_3(float* __restrict__ T_divide, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  T_divide[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = ((ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] / ph_3[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]) / ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(8) default_function_kernel_1(float* __restrict__ compute, float* __restrict__ ph_0, float* __restrict__ ph_6) {
  float T_batch_matmul_NN[4];
  __shared__ float ph_6_shared[4];
  for (int i_outer_inner_init = 0; i_outer_inner_init < 4; ++i_outer_inner_init) {
    T_batch_matmul_NN[i_outer_inner_init] = 0.000000e+00f;
  }
  for (int k_outer_outer = 0; k_outer_outer < 5; ++k_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 4) {
      ph_6_shared[((int)threadIdx.x)] = ph_6[(((((((int)blockIdx.x) / 5) * 100) + (((int)threadIdx.x) * 25)) + (k_outer_outer * 5)) + (((int)blockIdx.x) % 5))];
    }
    __syncthreads();
    for (int i_outer_inner = 0; i_outer_inner < 4; ++i_outer_inner) {
      T_batch_matmul_NN[i_outer_inner] = (T_batch_matmul_NN[i_outer_inner] + (ph_0[(((((((int)blockIdx.x) / 5) * 160) + (((int)threadIdx.x) * 20)) + (i_outer_inner * 5)) + k_outer_outer)] * ph_6_shared[(((int)threadIdx.x) >> 1)]));
    }
  }
  for (int i1_inner = 0; i1_inner < 4; ++i1_inner) {
    compute[(((((((int)blockIdx.x) / 5) * 160) + (((int)threadIdx.x) * 20)) + (i1_inner * 5)) + (((int)blockIdx.x) % 5))] = acoshf(T_batch_matmul_NN[i1_inner]);
  }
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = __expf(ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_4(float* __restrict__ T_subtract, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  T_subtract[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = ((ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] / ph_3[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]) - ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel_2(float* __restrict__ compute, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  compute[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = __cosf((ph_0[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] / ph_3[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]));
}

