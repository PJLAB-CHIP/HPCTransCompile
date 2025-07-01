extern "C" __global__ void __launch_bounds__(8) default_function_kernel_2(float* __restrict__ T_reverse_sequence, float* __restrict__ T_softmax_maxelem) {
  T_softmax_maxelem[((int)threadIdx.x)] = -3.402823e+38f;
  for (int k = 0; k < 21; ++k) {
    T_softmax_maxelem[((int)threadIdx.x)] = max(T_softmax_maxelem[((int)threadIdx.x)], T_reverse_sequence[((((int)threadIdx.x) * 21) + k)]);
  }
}

extern "C" __global__ void __launch_bounds__(8) default_function_kernel_4(float* __restrict__ T_reverse_sequence, float* __restrict__ T_softmax_maxelem) {
  T_softmax_maxelem[((int)threadIdx.x)] = 0.000000e+00f;
  for (int k = 0; k < 21; ++k) {
    T_softmax_maxelem[((int)threadIdx.x)] = (T_softmax_maxelem[((int)threadIdx.x)] + T_reverse_sequence[((((int)threadIdx.x) * 21) + k)]);
  }
}

extern "C" __global__ void __launch_bounds__(765) default_function_kernel_1(float* __restrict__ T_reverse_sequence, float* __restrict__ T_strided_slice) {
  if (((int)threadIdx.x) < 168) {
    T_reverse_sequence[((int)threadIdx.x)] = T_strided_slice[(((((((int)threadIdx.x) / 21) * 21) + (((int)threadIdx.x) % 7)) + 14) - (((((int)threadIdx.x) % 21) / 7) * 7))];
  }
}

extern "C" __global__ void __launch_bounds__(168) default_function_kernel_3(float* __restrict__ T_reverse_sequence, float* __restrict__ T_softmax_maxelem) {
  T_reverse_sequence[((int)threadIdx.x)] = __expf((T_reverse_sequence[((int)threadIdx.x)] - T_softmax_maxelem[(((int)threadIdx.x) / 21)]));
}

extern "C" __global__ void __launch_bounds__(765) default_function_kernel(float* __restrict__ T_strided_slice, float* __restrict__ ph) {
  if (((int)threadIdx.x) < 168) {
    T_strided_slice[((int)threadIdx.x)] = ph[(((((((int)threadIdx.x) / 21) * 60) + (((((int)threadIdx.x) % 21) / 7) * 12)) + (((int)threadIdx.x) % 7)) + 85)];
  }
}

extern "C" __global__ void __launch_bounds__(168) default_function_kernel_5(float* __restrict__ T_reverse_sequence, float* __restrict__ T_softmax_maxelem, float* __restrict__ T_softmax_norm) {
  T_softmax_norm[((int)threadIdx.x)] = (T_reverse_sequence[((int)threadIdx.x)] / T_softmax_maxelem[(((int)threadIdx.x) / 21)]);
}

