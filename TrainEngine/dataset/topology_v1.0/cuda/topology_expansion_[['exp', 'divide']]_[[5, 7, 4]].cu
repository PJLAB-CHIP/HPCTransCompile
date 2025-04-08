extern "C" __global__ void __launch_bounds__(40) default_function_kernel(float* __restrict__ T_divide, float* __restrict__ ph_0) {
  T_divide[((((int)blockIdx.x) * 40) + ((int)threadIdx.x))] = (ph_0[((((int)blockIdx.x) * 40) + ((int)threadIdx.x))] / __expf(ph_0[((((int)blockIdx.x) * 40) + ((int)threadIdx.x))]));
}

