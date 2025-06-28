extern "C" __global__ void __launch_bounds__(36) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 36) + ((int)threadIdx.x))] = atanhf(fmodf(ph_0[((((int)blockIdx.x) * 36) + ((int)threadIdx.x))], __cosf(ph_0[((((int)blockIdx.x) * 36) + ((int)threadIdx.x))])));
}

