extern "C" __global__ void __launch_bounds__(40) default_function_kernel(float* __restrict__ output_unpack, float* __restrict__ packed_out) {
  output_unpack[((((int)blockIdx.x) * 40) + ((int)threadIdx.x))] = packed_out[((((((int)blockIdx.x) * 40) + ((((int)threadIdx.x) / 20) * 20)) + ((((int)threadIdx.x) % 10) * 2)) + ((((int)threadIdx.x) % 20) / 10))];
}

