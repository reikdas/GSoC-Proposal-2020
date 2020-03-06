#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <assert.h>
#include <chrono>

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
  sub<T, C> <<<block, thread>>>(d_output, d_arr, d_arr2, startsoffset, stopsoffset, length);
  cudaDeviceSynchronize();
  thrust::device_vector<T> data(d_output, d_output + length);
  thrust::device_vector<T> temp(data.size() + 1);
  thrust::exclusive_scan(data.begin(), data.end(), temp.begin());
  temp[data.size()] = data.back() + temp[data.size() - 1];
  thrust::copy(temp.begin(), temp.end(), output);
  cudaFree(d_output);
  cudaFree(d_arr);
  cudaFree(d_arr2);
}

template <typename C, typename T>
void foo(T* tooffsets, const C* fromstarts, const C* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length) {
  tooffsets[0] = 0;
  for (int64_t i = 0; i < length; i++) {
    C start = fromstarts[startsoffset + i];
    C stop = fromstops[stopsoffset + i];
    assert(start <= stop);
    tooffsets[i + 1] = tooffsets[i] + (stop - start);
  }
}

template <typename T>
bool compare(T* arr1, T* arr2, int n) {
  for (int i=0; i<n; i++) {
    if (arr1[i] != arr2[i]) return false;
  }
  return true;
}

int main() {
  int const size = 60000;
  int starter[size], stopper[size], output[size + 1], output2[size + 1];
  for (int i = 0; i < size; i++) {
    starter[i] = i;
    stopper[i] = i + 1;
  }
  prefix_sum<int, int>(output, starter, stopper, 0, 0, size); // GPU Warm up
  cudaDeviceSynchronize();
  auto start1 = std::chrono::high_resolution_clock::now();
  prefix_sum<int, int>(output, starter, stopper, 0, 0, size);
  cudaDeviceSynchronize();
  auto stop1 = std::chrono::high_resolution_clock::now();
  auto time1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
  std::cout << "Time taken for GPU = " << time1.count() << "\n";
  auto start2 = std::chrono::high_resolution_clock::now();
  foo<int, int>(output2, starter, stopper, 0, 0, size);
  auto stop2 = std::chrono::high_resolution_clock::now();
  auto time2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);
  std::cout << "Time taken for CPU = " << time2.count() << "\n";
  for (int i=0; i<size; i++) {
    if (output2[i] != output[i]) {
      std::cout << "FALSE" << std::endl;
      return 0;
    }
  }
  return 0;
}
