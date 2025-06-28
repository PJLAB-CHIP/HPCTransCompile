extern "C" __global__ void __launch_bounds__(35) default_function_kernel(float* __restrict__ data, float* __restrict__ depth_to_space) {
  depth_to_space[((((int)blockIdx.x) * 35) + ((int)threadIdx.x))] = data[(((((((((((int)blockIdx.x) * 5) + (((int)threadIdx.x) / 7)) / 48) * 364) + ((((((((int)blockIdx.x) * 35) + ((int)threadIdx.x)) % 112) >> 3) % 2) * 168)) + (((((((int)blockIdx.x) * 3) + ((int)threadIdx.x)) & 7) % 2) * 84)) + (((((((int)blockIdx.x) * 5) + (((int)threadIdx.x) / 7)) % 48) >> 4) * 28)) + ((((((((int)blockIdx.x) * 35) + ((int)threadIdx.x)) % 112) >> 3) / 2) * 4)) + ((((((int)blockIdx.x) * 3) + ((int)threadIdx.x)) & 7) / 2))];
}

