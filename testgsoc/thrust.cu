#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <assert.h>

template <typename T, typename C>
__global__
void sub(T* output, const C* starter, const C* stopper, int64_t startsoffset, int64_t stopsoffset, int64_t n) {
  int thid = threadIdx.x + blockIdx.x * blockDim.x;
  if (thid < n) {
    C start = starter[thid + startsoffset];
    C stop = stopper[thid + stopsoffset];
    assert(start <= stop);
    output[thid] = stop - start;
  }
}

template <typename T, typename C>
void prefix_sum(T* output, const C* arr, const C* arr2, int64_t startsoffset, int64_t stopsoffset, int64_t length) {
  int block, thread;
  if (length > 1024) {
    block = (length / 1024) + 1;
    thread = 1024;
  }
  else {
    thread = length;
    block = 1;
  }
  T* d_output;
  C* d_arr, * d_arr2;
  cudaMalloc((void**)&d_output, length * sizeof(T));
  cudaMalloc((void**)&d_arr, length * sizeof(C));
  cudaMemcpy(d_arr, arr, length * sizeof(C), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_arr2, length * sizeof(C));
  cudaMemcpy(d_arr2, arr2, length * sizeof(C), cudaMemcpyHostToDevice);
  sub<T, C><<<block, thread>>>(d_output, d_arr, d_arr2, startsoffset, stopsoffset, length);
  cudaDeviceSynchronize();
  thrust::device_vector<T> data(d_output, d_output+length);
  thrust::device_vector<T> temp(data.size() + 1);
  thrust::exclusive_scan(data.begin(), data.end(), temp.begin());
  temp[data.size()] = data.back() + temp[data.size() - 1];
  thrust::copy(temp.begin(), temp.end(), output);
  cudaFree(d_output);
  cudaFree(d_arr);
  cudaFree(d_arr2);
}

int main() {
  int const size = 100000;
  int starter[size], stopper[size], output[size + 1];
  for (int i = 0; i < size; i++) {
    starter[i] = i;
    stopper[i] = i + 1;
  }
  prefix_sum<int, int>(output, starter, stopper, 0, 0, size);
  cudaDeviceSynchronize();
  for (int i = 0; i < size + 1; i++) {
    std::cout << output[i] << "\n";
  }
}
