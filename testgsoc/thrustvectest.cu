#include <iostream>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

template <typename T>
__global__
void sum(T* output, const T* start, const T* stop, const int n) {
  int thid = threadIdx.x + blockIdx.x * blockDim.x;
  if (thid < n) {
    output[thid] = stop[thid] - start[thid];
  }
}

template <typename T>
void prefix_sum1(T* output, const T* arr, const T* arr2, const int size) {
  int block, thread;
  if (size > 1024) {
    block = (size / 1024) + 1;
    thread = 1024;
  }
  else {
    thread = size;
    block = 1;
  }
  T* d_output, * d_arr, * d_arr2;
  cudaMalloc((void**)&d_output, size * sizeof(T));
  cudaMalloc((void**)&d_arr, size * sizeof(T));
  cudaMemcpy(d_arr, arr, size * sizeof(T), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_arr2, size * sizeof(T));
  cudaMemcpy(d_arr2, arr2, size * sizeof(T), cudaMemcpyHostToDevice);
  sum<T> << <block, thread >> > (d_output, d_arr, d_arr2, size);
  cudaDeviceSynchronize();
  cudaMemcpy(output, d_output, size * sizeof(T), cudaMemcpyDeviceToHost);
  thrust::device_vector<T> data(output, output+size);
  thrust::device_vector<T> temp(data.size() + 1);
  thrust::exclusive_scan(data.begin(), data.end(), temp.begin());
  temp[data.size()] = data.back() + temp[data.size() - 1];
  thrust::copy(temp.begin(), temp.end(), output);
  //for (const auto& i : temp)
  //  std::cout << i << '\n';
  cudaFree(d_output);
  cudaFree(d_arr);
  cudaFree(d_arr2);
}

template <typename T>
void prefix_sum2(T* output, const T* arr, const T* arr2, const int size) {
  thrust::device_vector<T> d_arr(arr, arr + size);
  thrust::device_vector<T> d_arr2(arr2, arr2 + size);
  thrust::device_vector<T> data(size);
  thrust::transform(d_arr2.begin(), d_arr2.end(), d_arr.begin(), data.begin(), thrust::minus<T>());
  thrust::device_vector<T> temp(data.size() + 1);
  thrust::exclusive_scan(data.begin(), data.end(), temp.begin());
  temp[data.size()] = data.back() + temp[data.size() - 1];
  thrust::copy(temp.begin(), temp.end(), output);
  //for (const auto& i : temp)
  //  std::cout << i << '\n';
}

int main() {
  int const size = 70000;
  int starter[size], stopper[size], output[size + 1];
  for (int i = 0; i < size; i++) {
    starter[i] = i;
    stopper[i] = i + 1;
  }
  prefix_sum1<int>(output, starter, stopper, size); // Warming up the GPU
  auto start1 = std::chrono::high_resolution_clock::now();
  prefix_sum1<int>(output, starter, stopper, size);
  auto stop1 = std::chrono::high_resolution_clock::now();
  auto time1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
  std::cout << "Time taken for kernel = " << time1.count() << "\n";
  auto start2 = std::chrono::high_resolution_clock::now();
  prefix_sum2<int>(output, starter, stopper, size);
  auto stop2 = std::chrono::high_resolution_clock::now();
  auto time2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);
  std::cout << "Time taken for thrust = " << time2.count() << "\n";
  //for (int i = 0; i < size + 1; i++)
  //  std::cout << output[i] << "\n";
}