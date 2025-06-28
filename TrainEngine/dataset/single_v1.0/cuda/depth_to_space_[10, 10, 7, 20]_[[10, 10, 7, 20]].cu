extern "C" __global__ void __launch_bounds__(50) default_function_kernel(float* __restrict__ data, float* __restrict__ depth_to_space) {
  depth_to_space[((((int)blockIdx.x) * 50) + ((int)threadIdx.x))] = data[(((((((((((int)blockIdx.x) * 5) + (((int)threadIdx.x) / 10)) / 112) * 1400) + ((((((((int)blockIdx.x) * 5) + (((int)threadIdx.x) / 10)) % 56) >> 2) % 2) * 560)) + (((((((int)blockIdx.x) * 10) + ((int)threadIdx.x)) % 40) % 2) * 280)) + (((((((int)blockIdx.x) * 5) + (((int)threadIdx.x) / 10)) % 112) / 56) * 140)) + ((((((((int)blockIdx.x) * 5) + (((int)threadIdx.x) / 10)) % 56) >> 2) / 2) * 20)) + ((((((int)blockIdx.x) * 10) + ((int)threadIdx.x)) % 40) / 2))];
}

