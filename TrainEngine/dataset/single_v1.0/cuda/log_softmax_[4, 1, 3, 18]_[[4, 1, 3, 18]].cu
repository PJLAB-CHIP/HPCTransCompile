extern "C" __global__ void __launch_bounds__(12) default_function_kernel_1(float* __restrict__ T_softmax_maxelem, float* __restrict__ compute, float* __restrict__ data) {
  compute[((int)threadIdx.x)] = 0.000000e+00f;
  for (int k = 0; k < 18; ++k) {
    compute[((int)threadIdx.x)] = (compute[((int)threadIdx.x)] + __expf((data[((((int)threadIdx.x) * 18) + k)] - T_softmax_maxelem[((int)threadIdx.x)])));
  }
}

extern "C" __global__ void __launch_bounds__(18) default_function_kernel_2(float* __restrict__ T_softmax_maxelem, float* __restrict__ compute, float* __restrict__ compute_1, float* __restrict__ data) {
  compute[((((int)blockIdx.x) * 18) + ((int)threadIdx.x))] = ((data[((((int)blockIdx.x) * 18) + ((int)threadIdx.x))] - T_softmax_maxelem[((int)blockIdx.x)]) - __logf(compute_1[((int)blockIdx.x)]));
}

extern "C" __global__ void __launch_bounds__(12) default_function_kernel(float* __restrict__ T_softmax_maxelem, float* __restrict__ data) {
  T_softmax_maxelem[((int)threadIdx.x)] = -3.402823e+38f;
  for (int k = 0; k < 18; ++k) {
    T_softmax_maxelem[((int)threadIdx.x)] = max(T_softmax_maxelem[((int)threadIdx.x)], data[((((int)threadIdx.x) * 18) + k)]);
  }
}

