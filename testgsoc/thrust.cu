#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

template <typename T>
void prefix_sum(T* output, const T* arr, const T* arr2, const int size) {
  thrust::device_vector<T> d_arr(arr, arr + size);
  thrust::device_vector<T> d_arr2(arr2, arr2 + size);
  thrust::device_vector<T> data(size);
  thrust::transform(d_arr2.begin(), d_arr2.end(), d_arr.begin(), data.begin(), thrust::minus<T>());
  thrust::device_vector<T> temp(data.size() + 1);
  thrust::exclusive_scan(data.begin(), data.end(), temp.begin());
  temp[data.size()] = data.back() + temp[data.size() - 1];
  thrust::copy(temp.begin(), temp.end(), output);
}

int main() {
  int const size = 70000;
  int starter[size], stopper[size], output[size + 1];
  for (int i = 0; i < size; i++) {
    starter[i] = i;
    stopper[i] = i + 1;
  }
  prefix_sum<int>(output, starter, stopper, size);
  for (int i = 0; i < size + 1; i++) {
    std::cout << output[i] << "\n";
  }
}