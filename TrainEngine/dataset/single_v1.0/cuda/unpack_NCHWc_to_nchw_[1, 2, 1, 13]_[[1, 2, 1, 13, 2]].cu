extern "C" __global__ void __launch_bounds__(4) default_function_kernel(float* __restrict__ output_unpack, float* __restrict__ packed_out) {
  output_unpack[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))] = packed_out[((((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 1)) / 13) * 26) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 13) * 2)) + ((((((int)blockIdx.x) * 4) + ((int)threadIdx.x)) % 26) / 13))];
}

