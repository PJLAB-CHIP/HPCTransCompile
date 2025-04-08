extern "C" __global__ void __launch_bounds__(4) default_function_kernel_1(float* __restrict__ T_subtract, float* __restrict__ ph_0) {
  T_subtract[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))] = (ph_0[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))] - acoshf(__expf(ph_0[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))])));
}

extern "C" __global__ void __launch_bounds__(64) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = ceilf(ph_0[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))]);
}

