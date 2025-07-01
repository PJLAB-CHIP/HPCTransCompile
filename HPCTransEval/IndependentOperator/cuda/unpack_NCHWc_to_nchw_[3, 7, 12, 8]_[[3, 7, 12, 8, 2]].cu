extern "C" __global__ void __launch_bounds__(64) default_function_kernel(float* __restrict__ output_unpack, float* __restrict__ packed_out) {
  output_unpack[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = packed_out[(((((((int)blockIdx.x) / 3) * 192) + ((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 3)) % 12) * 16)) + ((((int)threadIdx.x) & 7) * 2)) + ((((((int)blockIdx.x) % 3) * 2) + (((int)threadIdx.x) >> 5)) / 3))];
}

