extern "C" __global__ void __launch_bounds__(10) default_function_kernel(float* __restrict__ T_subtract, float* __restrict__ ph_0) {
  T_subtract[((((int)blockIdx.x) * 10) + ((int)threadIdx.x))] = (ph_0[((((int)blockIdx.x) * 10) + ((int)threadIdx.x))] - __expf(ph_0[((((int)blockIdx.x) * 10) + ((int)threadIdx.x))]));
}

