extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_5(float* __restrict__ T_broadcast_to) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 1035) {
    T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = min(9.000000e+00f, T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_7(float* __restrict__ T_broadcast_to, float* __restrict__ T_fast_tanh) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 1035) {
    T_fast_tanh[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ((T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * (((T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * (((T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * (((T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * (((T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * (((T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * (((T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * -2.760768e-16f) + 2.000188e-13f)) + -8.604672e-11f)) + 5.122297e-08f)) + 1.485722e-05f)) + 6.372619e-04f)) + 4.893525e-03f)) / (((T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * (((T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * (((T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) * 1.198258e-06f) + 1.185347e-04f)) + 2.268435e-03f)) + 4.893525e-03f));
  }
}

extern "C" __global__ void __launch_bounds__(69) default_function_kernel_3(float* __restrict__ T_broadcast_to, float* __restrict__ T_softmax_maxelem, float* __restrict__ compute) {
  compute[((int)threadIdx.x)] = 0.000000e+00f;
  for (int k = 0; k < 60; ++k) {
    compute[((int)threadIdx.x)] = (compute[((int)threadIdx.x)] + __expf((T_broadcast_to[((((int)threadIdx.x) * 60) + k)] - T_softmax_maxelem[((int)threadIdx.x)])));
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ T_reshape, float* __restrict__ ph) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 345) {
    T_reshape[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ph[(((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 15) * 60) + (((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 60))];
  }
}

extern "C" __global__ void __launch_bounds__(69) default_function_kernel_2(float* __restrict__ T_broadcast_to, float* __restrict__ T_softmax_maxelem) {
  T_softmax_maxelem[((int)threadIdx.x)] = -3.402823e+38f;
  for (int k = 0; k < 60; ++k) {
    T_softmax_maxelem[((int)threadIdx.x)] = max(T_softmax_maxelem[((int)threadIdx.x)], T_broadcast_to[((((int)threadIdx.x) * 60) + k)]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_1(float* __restrict__ T_broadcast_to, float* __restrict__ T_reshape) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 1035) {
    T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = T_reshape[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 1380)];
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_4(float* __restrict__ T_broadcast_to, float* __restrict__ T_softmax_maxelem, float* __restrict__ compute) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 1035) {
    T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ((T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] - T_softmax_maxelem[(((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 15)]) - __logf(compute[(((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 15)]));
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_6(float* __restrict__ T_broadcast_to) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 1035) {
    T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max(-9.000000e+00f, T_broadcast_to[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}

