#include <cuda_runtime.h>
#include <cassert>
#include "helper_cuda.h"
#include "device_launch_parameters.h"
#include <iostream>

template <typename C, typename T>
__global__
void awkward_listarray_compact_offsets(T* tooffsets, const C* fromstarts, const C* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length) {
  __shared__ int flag[1];
	int idx = threadIdx.x + (blockIdx.x * blockDim.x);
  if (idx == 0) {
    tooffsets[0] = 0;
    flag[1] = 0;
  }
  if (idx < length) {
    if (idx == 0) {
      for (int i = 0; i < length; i++) {
        C start = fromstarts[startsoffset + i];
        C stop = fromstops[stopsoffset + i];
        if (stop < start) {
          flag[0] = 1;
        }
        tooffsets[i + 1] = tooffsets[i] + (stop - start);
      }
    }
  }
  __syncthreads();
  if (flag[0] == 1) {
    printf("%d and %d\n", idx, flag[0]);
  }
}

template <typename T, typename C>
void offload(T* tooffsets, const C* fromstarts, const C* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length) {
  int* d_tooffsets, * d_fromstarts, * d_fromstops;
  checkCudaErrors(cudaMalloc((void**)&d_tooffsets, (length + 1) * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**)&d_fromstarts, length * sizeof(int)));
  checkCudaErrors(cudaMemcpy(d_fromstarts, fromstarts, length * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void**)&d_fromstops, length * sizeof(int)));
  checkCudaErrors(cudaMemcpy(d_fromstops, fromstops, length * sizeof(int), cudaMemcpyHostToDevice));
  int block, thread;
  if (length > 1024) {
    block = (length / 1024) + 1;
    thread = 1024;
  }
  else {
    thread = length;
    block = 1;
  }
  awkward_listarray_compact_offsets<int, int><<<block, thread, sizeof(int)>>>(d_tooffsets, d_fromstarts, d_fromstops, startsoffset, stopsoffset, length);
  cudaDeviceSynchronize();
  cudaMemcpy(tooffsets, d_tooffsets, (length + 1) * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_tooffsets);
  cudaFree(d_fromstarts);
  cudaFree(d_fromstops);
}

int main() {
  const int size = 10000;
  int tooffsets[size + 1], fromstarts[size], fromstops[size];
  for (int i = 0; i < size; i++) {
    fromstarts[i] = i + 100;
    fromstops[i] = i + 10;
  }
  offload<int, int>(tooffsets, fromstarts, fromstops, 0, 0, size);
  //for (int i = 0; i < size + 1; i++) {
  //  std::cout << tooffsets[i] << "\n";
  //}
  return 0;
}
