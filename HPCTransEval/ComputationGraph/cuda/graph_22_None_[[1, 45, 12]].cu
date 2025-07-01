extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_1(float* __restrict__ T_broadcast_to, float* __restrict__ T_transpose) {
  if (((int)threadIdx.x) < 75) {
    T_broadcast_to[((int)threadIdx.x)] = T_transpose[(((int)threadIdx.x) / 3)];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_2(float* __restrict__ T_broadcast_to) {
  if (((int)threadIdx.x) < 75) {
    T_broadcast_to[((int)threadIdx.x)] = floorf((T_broadcast_to[((int)threadIdx.x)] * 3.333333e-01f));
  }
}

extern "C" __global__ void __launch_bounds__(75) default_function_kernel_4(float* __restrict__ T_broadcast_to, float* __restrict__ T_strided_slice) {
  T_strided_slice[((int)threadIdx.x)] = T_broadcast_to[((int)threadIdx.x)];
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_3(float* __restrict__ T_broadcast_to) {
  if (((int)threadIdx.x) < 75) {
    T_broadcast_to[((int)threadIdx.x)] = ((max(min(T_broadcast_to[((int)threadIdx.x)], 4.000000e+00f), -4.000000e+00f) * (((max(min(T_broadcast_to[((int)threadIdx.x)], 4.000000e+00f), -4.000000e+00f) * max(min(T_broadcast_to[((int)threadIdx.x)], 4.000000e+00f), -4.000000e+00f)) * (((max(min(T_broadcast_to[((int)threadIdx.x)], 4.000000e+00f), -4.000000e+00f) * max(min(T_broadcast_to[((int)threadIdx.x)], 4.000000e+00f), -4.000000e+00f)) * (((max(min(T_broadcast_to[((int)threadIdx.x)], 4.000000e+00f), -4.000000e+00f) * max(min(T_broadcast_to[((int)threadIdx.x)], 4.000000e+00f), -4.000000e+00f)) * (((max(min(T_broadcast_to[((int)threadIdx.x)], 4.000000e+00f), -4.000000e+00f) * max(min(T_broadcast_to[((int)threadIdx.x)], 4.000000e+00f), -4.000000e+00f)) * (((max(min(T_broadcast_to[((int)threadIdx.x)], 4.000000e+00f), -4.000000e+00f) * max(min(T_broadcast_to[((int)threadIdx.x)], 4.000000e+00f), -4.000000e+00f)) * (((max(min(T_broadcast_to[((int)threadIdx.x)], 4.000000e+00f), -4.000000e+00f) * max(min(T_broadcast_to[((int)threadIdx.x)], 4.000000e+00f), -4.000000e+00f)) * -2.726142e-10f) + 2.770681e-08f)) + -2.101024e-06f)) + -5.692506e-05f)) + -7.349906e-04f)) + -2.954600e-03f)) + -1.609603e-02f)) / (((max(min(T_broadcast_to[((int)threadIdx.x)], 4.000000e+00f), -4.000000e+00f) * max(min(T_broadcast_to[((int)threadIdx.x)], 4.000000e+00f), -4.000000e+00f)) * (((max(min(T_broadcast_to[((int)threadIdx.x)], 4.000000e+00f), -4.000000e+00f) * max(min(T_broadcast_to[((int)threadIdx.x)], 4.000000e+00f), -4.000000e+00f)) * (((max(min(T_broadcast_to[((int)threadIdx.x)], 4.000000e+00f), -4.000000e+00f) * max(min(T_broadcast_to[((int)threadIdx.x)], 4.000000e+00f), -4.000000e+00f)) * (((max(min(T_broadcast_to[((int)threadIdx.x)], 4.000000e+00f), -4.000000e+00f) * max(min(T_broadcast_to[((int)threadIdx.x)], 4.000000e+00f), -4.000000e+00f)) * -1.456607e-05f) + -2.133740e-04f)) + -1.682827e-03f)) + -7.373329e-03f)) + -1.426474e-02f));
  }
}

extern "C" __global__ void __launch_bounds__(540) default_function_kernel(float* __restrict__ T_transpose, float* __restrict__ ph) {
  if (((int)threadIdx.x) < 25) {
    T_transpose[((int)threadIdx.x)] = ph[((((((int)threadIdx.x) / 5) * 12) + (((int)threadIdx.x) % 5)) + 14)];
  }
}

