extern "C" __global__ void __launch_bounds__(9) default_function_kernel(float* __restrict__ compute, float* __restrict__ ph_0) {
  compute[((((int)blockIdx.x) * 9) + ((int)threadIdx.x))] = __expf((ph_0[((((int)blockIdx.x) * 9) + ((int)threadIdx.x))] * asinhf(ph_0[((((int)blockIdx.x) * 9) + ((int)threadIdx.x))])));
}

