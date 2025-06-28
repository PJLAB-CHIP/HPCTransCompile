extern "C" __global__ void __launch_bounds__(49) default_function_kernel(float* __restrict__ Scale, float* __restrict__ compute, float* __restrict__ data) {
  compute[((((int)blockIdx.x) * 49) + ((int)threadIdx.x))] = ((0.000000e+00f < data[((((int)blockIdx.x) * 49) + ((int)threadIdx.x))]) ? data[((((int)blockIdx.x) * 49) + ((int)threadIdx.x))] : (data[((((int)blockIdx.x) * 49) + ((int)threadIdx.x))] * Scale[(((((int)blockIdx.x) * 7) + ((int)threadIdx.x)) % 14)]));
}

