#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>

template <typename C, typename T>
__global__
void awkward_listarray_compact_offsets(T* tooffsets, const C* fromstarts, const C* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length) {
	int idx = threadIdx.x + (blockIdx.x * blockDim.x);
  if (idx == 0) tooffsets[0] = 0;
  if (idx < length) {
    if (idx == 0) {
      for (int i = 0; i < length; i++) {
        C start = fromstarts[startsoffset + i];
        C stop = fromstops[stopsoffset + i];
        assert(start < stop);
        tooffsets[i + 1] = tooffsets[i] + (stop - start);
      }
    }
  }
}

template <typename T, typename C>
void offload(T* tooffsets, const C* fromstarts, const C* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length) {
  int* d_tooffsets, * d_fromstarts, * d_fromstops;
  cudaMalloc((void**)&d_tooffsets, (length + 1) * sizeof(int));
  cudaMalloc((void**)&d_fromstarts, length * sizeof(int));
  cudaMemcpy(d_fromstarts, fromstarts, length * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_fromstops, length * sizeof(int));
  cudaMemcpy(d_fromstops, fromstops, length * sizeof(int), cudaMemcpyHostToDevice);
  int block, thread;
  if (length > 1024) {
    block = (length / 1024) + 1;
    thread = 1024;
  }
  else {
    thread = length;
    block = 1;
  }
  awkward_listarray_compact_offsets <int, int> << <block, thread >> > (d_tooffsets, d_fromstarts, d_fromstops, startsoffset, stopsoffset, length);
  cudaDeviceSynchronize();
  cudaMemcpy(tooffsets, d_tooffsets, (length + 1) * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_tooffsets);
  cudaFree(d_fromstarts);
  cudaFree(d_fromstops);
}

int main() {
  const int size = 10;
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