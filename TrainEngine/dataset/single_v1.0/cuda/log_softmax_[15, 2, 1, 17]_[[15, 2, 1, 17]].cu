extern "C" __global__ void __launch_bounds__(30) default_function_kernel(float* __restrict__ T_softmax_maxelem, float* __restrict__ data) {
  T_softmax_maxelem[((int)threadIdx.x)] = -3.402823e+38f;
  for (int k = 0; k < 17; ++k) {
    T_softmax_maxelem[((int)threadIdx.x)] = max(T_softmax_maxelem[((int)threadIdx.x)], data[((((int)threadIdx.x) * 17) + k)]);
  }
}

extern "C" __global__ void __launch_bounds__(24) default_function_kernel_2(float* __restrict__ T_softmax_maxelem, float* __restrict__ compute, float* __restrict__ compute_1, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) / 6)) < 85) {
    compute[((((int)blockIdx.x) * 24) + ((int)threadIdx.x))] = ((data[((((int)blockIdx.x) * 24) + ((int)threadIdx.x))] - T_softmax_maxelem[(((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) / 17)]) - __logf(compute_1[(((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) / 17)]));
  }
}

extern "C" __global__ void __launch_bounds__(30) default_function_kernel_1(float* __restrict__ T_softmax_maxelem, float* __restrict__ compute, float* __restrict__ data) {
  compute[((int)threadIdx.x)] = 0.000000e+00f;
  for (int k = 0; k < 17; ++k) {
    compute[((int)threadIdx.x)] = (compute[((int)threadIdx.x)] + __expf((data[((((int)threadIdx.x) * 17) + k)] - T_softmax_maxelem[((int)threadIdx.x)])));
  }
}

