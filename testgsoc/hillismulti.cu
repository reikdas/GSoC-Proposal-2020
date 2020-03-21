#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>

__global__
void prefix_sum(int iter, int* tooffsets, const int* input, int64_t length, int loader) {
	int thid = threadIdx.x;
  extern __shared__ int temp[];
  int pout = 0, pin = 1;
  if (thid < length) {
    temp[thid] = input[thid + (iter*1024)];
    __syncthreads();
    for (int offset = 1; offset < length; offset *=2) {
      pout = 1 - pout;
      pin = 1 - pout;
      if (thid >= offset)
        temp[pout*length + thid] = temp[pin*length + thid - offset] + temp[pin*length + thid];
      else
        temp[pout*length + thid] = temp[pin*length + thid];
      __syncthreads();
    }
    tooffsets[thid] = temp[pout*length + thid] + loader;
  }
}

void offload(int* tooffsets, const int* input, int64_t length) {
  int* d_tooffsets;
  int* d_input;
  cudaMalloc((void**)&d_tooffsets, 1024 * sizeof(int));
  cudaMalloc((void**)&d_input, length * sizeof(int));
  cudaMemcpy(d_input, input, length * sizeof(int), cudaMemcpyHostToDevice);
  int block, thread;
  if (length > 1024) {
    block = (length / 1024) + 1;
    thread = 1024;
  }
  else {
    thread = length;
    block = 1;
  }
  int sums = 0;
  for (int i = 0; i < block; i++) {
    prefix_sum<<<1, thread, thread*2*sizeof(int)>>>(i, d_tooffsets, d_input, thread, sums);
    cudaDeviceSynchronize();
    int temp1;
    if (((i+1)*1024) > length ) {
      temp1 = length%1024;
    }
    else {
      temp1 = 1024;
    }
    cudaMemcpy(tooffsets+(i*1024), d_tooffsets, temp1 * sizeof(int), cudaMemcpyDeviceToHost);
    sums = tooffsets[(i*1024) + temp1 - 1];
  }
  tooffsets[length] = tooffsets[length - 1] + input[length - 1];
  cudaFree(d_tooffsets);
  cudaFree(d_input);
}

int main() {
  const int size = 600000;
  int tooffsets[size + 1], input[size];
  for (int i = 0; i < size; i++) {
    input[i] = 10;
  }
  offload(tooffsets, input, size);
  for (int i = 0; i < size + 1; i++) {
	  std::cout << tooffsets[i] << "\n";
  }
  return 0;
}
