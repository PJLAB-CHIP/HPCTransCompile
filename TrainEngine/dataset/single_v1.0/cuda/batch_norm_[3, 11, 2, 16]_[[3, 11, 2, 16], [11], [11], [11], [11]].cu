extern "C" __global__ void __launch_bounds__(44) default_function_kernel(float* __restrict__ T_divide, float* __restrict__ data, float* __restrict__ moving_mean, float* __restrict__ moving_var) {
  T_divide[((((int)blockIdx.x) * 44) + ((int)threadIdx.x))] = ((data[((((int)blockIdx.x) * 44) + ((int)threadIdx.x))] - moving_mean[((((((int)blockIdx.x) & 7) * 11) + (((int)threadIdx.x) >> 2)) >> 3)]) / sqrtf((moving_var[((((((int)blockIdx.x) & 7) * 11) + (((int)threadIdx.x) >> 2)) >> 3)] + 1.000000e-05f)));
}

