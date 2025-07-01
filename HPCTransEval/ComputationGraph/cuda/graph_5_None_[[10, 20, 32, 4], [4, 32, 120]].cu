extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_3(float* __restrict__ T_transpose) {
  T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (1.100000e+01f - T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_1(float* __restrict__ T_reshape, float* __restrict__ T_transpose) {
  T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_reshape[((((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) % 120) * 128) + (((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) >> 3)) % 480) / 15) * 4)) + (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) / 15))];
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ T_reshape, float* __restrict__ ph) {
  T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ph[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))];
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_2(float* __restrict__ T_transpose) {
  T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + 3.000000e+00f);
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_4(float* __restrict__ T_transpose, float* __restrict__ ph) {
  T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * ph[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_5(float* __restrict__ T_power, float* __restrict__ T_transpose) {
  T_power[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = powf(T_transpose[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 2.000000e+00f);
}

