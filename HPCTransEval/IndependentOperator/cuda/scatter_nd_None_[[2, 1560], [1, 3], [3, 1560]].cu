extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_1(float* __restrict__ atomic_add_return, int* __restrict__ indices, float* __restrict__ scatter_nd_cuda, float* __restrict__ updates) {
  if (((((int)blockIdx.y) * 128) + (((int)threadIdx.x) >> 3)) < 195) {
    atomic_add_return[0] = atomicAdd((&(scatter_nd_cuda[(((indices[((int)blockIdx.x)] * 1560) + (((int)blockIdx.y) * 1024)) + ((int)threadIdx.x))])), updates[(((((int)blockIdx.x) * 1560) + (((int)blockIdx.y) * 1024)) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ data, float* __restrict__ scatter_nd_cuda) {
  if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) < 195) {
    scatter_nd_cuda[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = data[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))];
  }
}

