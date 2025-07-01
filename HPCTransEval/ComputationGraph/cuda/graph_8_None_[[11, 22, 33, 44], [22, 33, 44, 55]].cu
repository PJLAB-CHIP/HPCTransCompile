extern "C" __global__ void __launch_bounds__(1000) default_function_kernel_3(float* __restrict__ T_strided_slice, float* __restrict__ T_strided_slice_1) {
  T_strided_slice[((int)threadIdx.x)] = -3.402823e+38f;
  for (int k = 0; k < 10; ++k) {
    T_strided_slice[((int)threadIdx.x)] = max(T_strided_slice[((int)threadIdx.x)], T_strided_slice_1[((((((int)threadIdx.x) / 10) * 100) + (k * 10)) + (((int)threadIdx.x) % 10))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_1(float* __restrict__ T_strided_slice, float* __restrict__ ph) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 625) {
    T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ph[((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) >> 3)) / 125) * 79860) + (((((((int)blockIdx.x) * 6) + (((int)threadIdx.x) >> 2)) % 250) / 25) * 2420)) + (((((((int)blockIdx.x) * 12) + (((int)threadIdx.x) >> 1)) % 50) / 5) * 55)) + (((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 10)) + 933559)];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ T_strided_slice, float* __restrict__ ph) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 625) {
    T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ph[((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) >> 3)) / 125) * 31944) + (((((((int)blockIdx.x) * 6) + (((int)threadIdx.x) >> 2)) % 250) / 25) * 1452)) + (((((((int)blockIdx.x) * 12) + (((int)threadIdx.x) >> 1)) % 50) / 5) * 44)) + (((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 10)) + 16973)];
  }
}

extern "C" __global__ void __launch_bounds__(1000) default_function_kernel_4(float* __restrict__ T_strided_slice, float* __restrict__ T_strided_slice_1, float* __restrict__ compute) {
  compute[((int)threadIdx.x)] = 0.000000e+00f;
  for (int k = 0; k < 10; ++k) {
    compute[((int)threadIdx.x)] = (compute[((int)threadIdx.x)] + __expf((T_strided_slice[((((((int)threadIdx.x) / 10) * 100) + (k * 10)) + (((int)threadIdx.x) % 10))] - T_strided_slice_1[((int)threadIdx.x)])));
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_5(float* __restrict__ T_strided_slice, float* __restrict__ T_strided_slice_1, float* __restrict__ compute, float* __restrict__ compute_1) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 625) {
    compute[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ((T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] - T_strided_slice_1[(((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 25) * 10) + (((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 10))]) - __logf(compute_1[(((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 25) * 10) + (((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 10))]));
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_2(float* __restrict__ T_strided_slice, float* __restrict__ T_strided_slice_1) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 625) {
    T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_strided_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + T_strided_slice_1[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

