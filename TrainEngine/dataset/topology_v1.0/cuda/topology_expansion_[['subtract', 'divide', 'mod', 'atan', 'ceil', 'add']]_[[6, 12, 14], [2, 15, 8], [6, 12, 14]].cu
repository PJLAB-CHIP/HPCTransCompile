extern "C" __global__ void __launch_bounds__(10) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  compute[((((int)blockIdx.x) * 10) + ((int)threadIdx.x))] = atanf(fmodf(ph_0[((((int)blockIdx.x) * 10) + ((int)threadIdx.x))], (ph_0[((((int)blockIdx.x) * 10) + ((int)threadIdx.x))] / ph_3[((((int)blockIdx.x) * 10) + ((int)threadIdx.x))])));
}

extern "C" __global__ void __launch_bounds__(8) default_function_kernel_1(float* __restrict__ compute, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  compute[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))] = ceilf(fmodf(ph_0[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))], (ph_0[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))] / ph_3[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))])));
}

extern "C" __global__ void __launch_bounds__(4) default_function_kernel_2(float* __restrict__ T_add, float* __restrict__ ph_0, float* __restrict__ ph_3) {
  T_add[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))] = ((ph_0[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))] - ph_3[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))]) + ph_0[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))]);
}

