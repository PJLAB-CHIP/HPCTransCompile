extern "C" __global__ void __launch_bounds__(20) default_function_kernel(float* __restrict__ T_multiply, float* __restrict__ ph_0) {
  T_multiply[((((int)blockIdx.x) * 20) + ((int)threadIdx.x))] = (ph_0[((((int)blockIdx.x) * 20) + ((int)threadIdx.x))] * ceilf(ph_0[((((int)blockIdx.x) * 20) + ((int)threadIdx.x))]));
}

