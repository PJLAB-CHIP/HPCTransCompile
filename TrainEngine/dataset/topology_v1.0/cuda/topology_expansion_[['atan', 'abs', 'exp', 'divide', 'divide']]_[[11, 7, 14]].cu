extern "C" __global__ void default_function_kernel_1(float* __restrict__ T_divide, float* __restrict__ ph_0) {
  T_divide[((int)blockIdx.x)] = (ph_0[((int)blockIdx.x)] / (ph_0[((int)blockIdx.x)] / __expf(fabsf(ph_0[((int)blockIdx.x)]))));
}

extern "C" __global__ void __launch_bounds__(20) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 20) + ((int)threadIdx.x))] = atanf(ph_0[((((int)blockIdx.x) * 20) + ((int)threadIdx.x))]);
}

