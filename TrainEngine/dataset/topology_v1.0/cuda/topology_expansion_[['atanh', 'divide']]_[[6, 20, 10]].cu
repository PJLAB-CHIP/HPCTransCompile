extern "C" __global__ void __launch_bounds__(5) default_function_kernel(float* __restrict__ T_divide, float* __restrict__ ph_0) {
  T_divide[((((int)blockIdx.x) * 5) + ((int)threadIdx.x))] = (ph_0[((((int)blockIdx.x) * 5) + ((int)threadIdx.x))] / atanhf(ph_0[((((int)blockIdx.x) * 5) + ((int)threadIdx.x))]));
}

