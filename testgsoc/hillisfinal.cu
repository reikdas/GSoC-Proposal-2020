#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>

template <typename T, typename C>
__global__
void awkward_listarray_compact_offsets(T* tooffsets, const C* fromstarts, const C* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length, T* sums) {
  int thid = threadIdx.x + (blockIdx.x * blockDim.x);
  extern __shared__ T temp[];
  int pout = 0, pin = 1;
  if (thid < length) {
    temp[thid] = fromstops[stopsoffset + thid] - fromstarts[startsoffset + thid];
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
    tooffsets[thid] = temp[pout*length + thid];
    __syncthreads();
    if ((thid == 1023) || ((blockIdx.x != 0) && (thid == ((1024 * (blockIdx.x + 1))-1))) || (thid == length-1)) {
        sums[blockIdx.x] = tooffsets[thid];
    }
  }
}

template <typename T>
__global__
void prefix_sum(T* tooffsets, int length) {
  int thid = threadIdx.x + (blockIdx.x * blockDim.x);
  extern __shared__ T temp[];
  int pout = 0, pin = 1;
  if (thid < length) {
    temp[thid] = tooffsets[thid];
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
    tooffsets[thid] = temp[pout*length + thid];
  }
}

template<typename T>
__global__
void adder(T* tooffsets, T* sums, int64_t length) {
  int thid = threadIdx.x + (blockIdx.x * blockDim.x);
  if (blockIdx.x != 0 && thid < length)
    tooffsets[thid] += sums[blockIdx.x - 1];
}

template <typename T, typename C>
void offload(T* tooffsets, const C* fromstarts, const C* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length) {
  T* d_tooffsets, * d_sums;
  C* d_fromstarts, * d_fromstops;
  cudaMalloc((void**)&d_tooffsets, (length+1) * sizeof(T));
  cudaMalloc((void**)&d_fromstarts, length * sizeof(C));
  cudaMemcpy(d_fromstarts, fromstarts, length * sizeof(C), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_fromstops, length * sizeof(C));
  cudaMemcpy(d_fromstops, fromstops, length * sizeof(C), cudaMemcpyHostToDevice);
  int block, thread;
  if (length > 1024) {
    block = (length / 1024) + 1;
    thread = 1024;
  }
  else {
    thread = length;
    block = 1;
  }
  cudaMalloc((void**)&d_sums, block*sizeof(T));
  awkward_listarray_compact_offsets<T, C><<<block, thread, length*2*sizeof(T)>>>(d_tooffsets, d_fromstarts, d_fromstops, startsoffset, stopsoffset, length, d_sums);
  cudaDeviceSynchronize();
  prefix_sum<T><<<1, block, block*2*sizeof(T)>>>(d_sums, block);
  cudaDeviceSynchronize();
  adder<T><<<block, thread>>>(d_tooffsets, d_sums, length);
  cudaDeviceSynchronize();
  cudaMemcpy(tooffsets, d_tooffsets, (length + 1) * sizeof(T), cudaMemcpyDeviceToHost);
  tooffsets[length] = tooffsets[length - 1] + fromstops[length - 1 + stopsoffset] - fromstarts[length - 1 + startsoffset];
  cudaFree(d_tooffsets);
  cudaFree(d_fromstarts);
  cudaFree(d_fromstops);
  cudaFree(d_sums);
}

int main() {
  const int size = 6000;
  int tooffsets[size + 1], fromstarts[size], fromstops[size];
  for (int i = 0; i < size; i++) {
    fromstarts[i] = i;
    fromstops[i] = i + 10;
  }
  offload<int, int>(tooffsets, fromstarts, fromstops, 0, 0, size);
  for (int i = 0; i < size + 1; i++) {
	  std::cout << tooffsets[i] << "\n";
  }
  return 0;
}