extern "C" __global__ void default_function_kernel_1(float* __restrict__ T_multiply, float* __restrict__ ph_0) {
  T_multiply[((int)blockIdx.x)] = (ph_0[((int)blockIdx.x)] * acoshf(atanf(ph_0[((int)blockIdx.x)])));
}

extern "C" __global__ void __launch_bounds__(10) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 10) + ((int)threadIdx.x))] = atanhf(ph_0[((((int)blockIdx.x) * 10) + ((int)threadIdx.x))]);
}

