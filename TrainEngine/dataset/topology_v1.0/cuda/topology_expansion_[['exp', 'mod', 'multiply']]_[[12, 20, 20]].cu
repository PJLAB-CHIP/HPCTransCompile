extern "C" __global__ void __launch_bounds__(8) default_function_kernel(float* __restrict__ T_multiply, float* __restrict__ ph_0) {
  T_multiply[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))] = (fmodf(ph_0[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))], __expf(ph_0[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))])) * ph_0[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))]);
}

