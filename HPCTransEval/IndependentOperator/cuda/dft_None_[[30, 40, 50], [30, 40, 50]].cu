extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ Im, float* __restrict__ Re, float* __restrict__ dft_cuda, float* __restrict__ dft_cuda_1) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 75) {
    for (int i = 0; i < 50; ++i) {
      dft_cuda[(((((int)blockIdx.x) * 51200) + (((int)threadIdx.x) * 50)) + i)] = 0.000000e+00f;
      dft_cuda_1[(((((int)blockIdx.x) * 51200) + (((int)threadIdx.x) * 50)) + i)] = 0.000000e+00f;
      for (int j = 0; j < 50; ++j) {
        float cse_var_1 = (((-6.283185e+00f * ((float)i)) * 2.000000e-02f) * ((float)j));
        dft_cuda[(((((int)blockIdx.x) * 51200) + (((int)threadIdx.x) * 50)) + i)] = (dft_cuda[(((((int)blockIdx.x) * 51200) + (((int)threadIdx.x) * 50)) + i)] + ((Re[(((((int)blockIdx.x) * 51200) + (((int)threadIdx.x) * 50)) + j)] * __cosf(cse_var_1)) - (Im[(((((int)blockIdx.x) * 51200) + (((int)threadIdx.x) * 50)) + j)] * __sinf(cse_var_1))));
        dft_cuda_1[(((((int)blockIdx.x) * 51200) + (((int)threadIdx.x) * 50)) + i)] = (dft_cuda_1[(((((int)blockIdx.x) * 51200) + (((int)threadIdx.x) * 50)) + i)] + ((Re[(((((int)blockIdx.x) * 51200) + (((int)threadIdx.x) * 50)) + j)] * __sinf(cse_var_1)) + (Im[(((((int)blockIdx.x) * 51200) + (((int)threadIdx.x) * 50)) + j)] * __cosf(cse_var_1))));
      }
    }
  }
}

