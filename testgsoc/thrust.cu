#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

template <typename T, typename C>
void prefix_sum(T* output, const C* arr, const C* arr2, int64_t startsoffset, int64_t stopsoffset, int64_t length) {
  thrust::device_vector<C> d_arr(arr + startsoffset, arr + startsoffset + length);
  thrust::device_vector<C> d_arr2(arr2 + stopsoffset, arr2 + stopsoffset + length);
  thrust::device_vector<C> data(length);
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
  prefix_sum<int, int>(output, starter, stopper, 0, 0, size);
  for (int i = 0; i < size + 1; i++) {
    std::cout << output[i] << "\n";
  }
}