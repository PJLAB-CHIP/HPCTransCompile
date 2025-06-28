extern "C" __global__ void __launch_bounds__(48) default_function_kernel(float* __restrict__ output_unpack, float* __restrict__ packed_out) {
  if (((((int)blockIdx.x) * 6) + (((int)threadIdx.x) >> 3)) < 175) {
    output_unpack[((((int)blockIdx.x) * 48) + ((int)threadIdx.x))] = packed_out[((((((((int)blockIdx.x) * 6) + (((int)threadIdx.x) >> 3)) / 25) * 200) + ((((((int)blockIdx.x) * 48) + ((int)threadIdx.x)) % 100) * 2)) + ((((((int)blockIdx.x) * 12) + (((int)threadIdx.x) >> 2)) % 50) / 25))];
  }
}

