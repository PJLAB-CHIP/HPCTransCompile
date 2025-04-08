extern "C" __global__ void __launch_bounds__(2) default_function_kernel_1(float* __restrict__ T_subtract, float* __restrict__ ph_0) {
  T_subtract[((((int)blockIdx.x) * 2) + ((int)threadIdx.x))] = (ph_0[((((int)blockIdx.x) * 2) + ((int)threadIdx.x))] - fabsf(ph_0[((((int)blockIdx.x) * 2) + ((int)threadIdx.x))]));
}

extern "C" __global__ void __launch_bounds__(10) default_function_kernel(float* __restrict__ T_multiply, float* __restrict__ ph_0) {
  T_multiply[((((int)blockIdx.x) * 10) + ((int)threadIdx.x))] = (fmodf(ph_0[((((int)blockIdx.x) * 10) + ((int)threadIdx.x))], acoshf(ph_0[((((int)blockIdx.x) * 10) + ((int)threadIdx.x))])) * ph_0[((((int)blockIdx.x) * 10) + ((int)threadIdx.x))]);
}

