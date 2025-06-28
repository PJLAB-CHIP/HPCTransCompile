extern "C" __global__ void default_function_kernel_1(float* __restrict__ T_mod, float* __restrict__ ph_0) {
  T_mod[((int)blockIdx.x)] = fmodf(ph_0[((int)blockIdx.x)], (ph_0[((int)blockIdx.x)] * atanf(__expf(ph_0[((int)blockIdx.x)]))));
}

extern "C" __global__ void __launch_bounds__(10) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 10) + ((int)threadIdx.x))] = asinf(ph_0[((((int)blockIdx.x) * 10) + ((int)threadIdx.x))]);
}

