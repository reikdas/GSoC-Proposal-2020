#include <iostream>
#include <fstream>
#include <thrust/device_vector.h> #include <thrust/scan.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <assert.h>
#include <chrono>
#include "helper_cuda.h"
#include <iomanip>

template <typename C, typename T>
__global__
void naivemultikernel(int block, T* tooffsets, const C* fromstarts, const C* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length) {
	int idx = threadIdx.x + (block*1024);
  if (idx < length) {
    if (idx == 0) tooffsets[0] = 0;
    for (int i = block*1024; i < std::min((int)length, (block+1)*1024); i++) {
      __syncthreads();
      if (i == idx) {
        C start = fromstarts[startsoffset + i];
        C stop = fromstops[stopsoffset + i];
        assert(start < stop);
        tooffsets[i + 1] = tooffsets[i] + (stop - start);
      }
    }
  }
}

template <typename T, typename C>
int naivemulti(T* tooffsets, const C* fromstarts, const C* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length) {
  int* d_tooffsets, * d_fromstarts, * d_fromstops;
  cudaMalloc((void**)&d_tooffsets, (length+1) * sizeof(T));
  cudaMalloc((void**)&d_fromstarts, length * sizeof(T));
  cudaMemcpy(d_fromstarts, fromstarts, length * sizeof(T), cudaMemcpyHostToDevice);
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
  auto start1 = std::chrono::high_resolution_clock::now();
  for (int i=0; i<block; i++) {
    naivemultikernel<T, C><<<1, thread>>>(i, d_tooffsets, d_fromstarts, d_fromstops, startsoffset, stopsoffset, length);
  }
  cudaDeviceSynchronize();
  auto stop1 = std::chrono::high_resolution_clock::now();
  auto time1 = std::chrono::duration_cast<std::chrono::nanoseconds>(stop1 - start1);
  cudaMemcpy(tooffsets, d_tooffsets, (length + 1) * sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(d_tooffsets);
  cudaFree(d_fromstarts);
  cudaFree(d_fromstops);
  return time1.count();
}

template <typename C, typename T>
__global__
void naivesinglekernel(T* tooffsets, const C* fromstarts, const C* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length) {
	int idx = threadIdx.x + (blockIdx.x * blockDim.x);
  if (idx < length) {
    if (idx == 0) {
      for (int i = 0; i < length; i++) {
        C start = fromstarts[startsoffset + i];
        C stop = fromstops[stopsoffset + i];
        assert (start <= stop);
        tooffsets[i + 1] = tooffsets[i] + (stop - start);
      }
    }
  }
}

template <typename T, typename C>
int naivesingle(T* tooffsets, const C* fromstarts, const C* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length) {
  int* d_tooffsets, * d_fromstarts, * d_fromstops;
  checkCudaErrors(cudaMalloc((void**)&d_tooffsets, (length + 1) * sizeof(T)));
  checkCudaErrors(cudaMalloc((void**)&d_fromstarts, length * sizeof(T)));
  checkCudaErrors(cudaMemcpy(d_fromstarts, fromstarts, length * sizeof(T), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void**)&d_fromstops, length * sizeof(C)));
  checkCudaErrors(cudaMemcpy(d_fromstops, fromstops, length * sizeof(C), cudaMemcpyHostToDevice));
  int block, thread;
  if (length > 1024) {
    block = (length / 1024) + 1;
    thread = 1024;
  }
  else {
    thread = length;
    block = 1;
  }
  auto start1 = std::chrono::high_resolution_clock::now();
  naivesinglekernel<T, C><<<block, thread, sizeof(T)>>>(d_tooffsets, d_fromstarts, d_fromstops, startsoffset, stopsoffset, length);
  cudaDeviceSynchronize();
  auto stop1 = std::chrono::high_resolution_clock::now();
  auto time1 = std::chrono::duration_cast<std::chrono::nanoseconds>(stop1 - start1);
  cudaDeviceSynchronize();
  cudaMemcpy(tooffsets, d_tooffsets, (length + 1) * sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(d_tooffsets);
  cudaFree(d_fromstarts);
  cudaFree(d_fromstops);
  return time1.count();
}

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
int gpupar(T* output, const C* arr, const C* arr2, int64_t startsoffset, int64_t stopsoffset, int64_t length) {
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
  auto start1 = std::chrono::high_resolution_clock::now();
  sub<T, C> <<<block, thread>>>(d_output, d_arr, d_arr2, startsoffset, stopsoffset, length);
  cudaDeviceSynchronize();
  auto stop1 = std::chrono::high_resolution_clock::now();
  auto time1 = std::chrono::duration_cast<std::chrono::nanoseconds>(stop1 - start1);
  auto start2 = std::chrono::high_resolution_clock::now();
  thrust::device_vector<T> data(d_output, d_output + length);
  thrust::device_vector<T> temp(data.size() + 1);
  thrust::exclusive_scan(data.begin(), data.end(), temp.begin());
  temp[data.size()] = data.back() + temp[data.size() - 1];
  auto stop2 = std::chrono::high_resolution_clock::now();
  auto time2 = std::chrono::duration_cast<std::chrono::nanoseconds>(stop1 - start1);
  thrust::copy(temp.begin(), temp.end(), output);
  cudaFree(d_output);
  cudaFree(d_arr);
  cudaFree(d_arr2);
  auto time = time1.count() + time2.count();
  return (int)time;
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

int main() {
  // Warm up GPU
  const int t_size = 1024;
  int t_starter[t_size], t_stopper[t_size], t_output[t_size + 1];
  for (int i = 0; i < t_size; i++) {
    t_starter[i] = i;
    t_stopper[i] = i + 1;
  }
  int throwaway = gpupar<int, int>(t_output, t_starter, t_stopper, 0, 0, t_size);
  // -----------------------------------------------------------
  std::ofstream outfile;
  outfile.open("data.txt");
  for (int counter=10; counter<600000; counter+=1000) {
    const int size = counter;
    std::cout << "Benchmark for array of size " << counter << "\n";
    outfile << counter;
    int starter[size], stopper[size], output[size + 1];
    for (int i = 0; i < size; i++) {
      starter[i] = i;
      stopper[i] = i + 1;
    }
    int tot = 0;
    double time = 0.00;
    for (int i = 0; i < 5; i++) {
     tot = tot + gpupar<int, int>(output, starter, stopper, 0, 0, size);
    }
    time = ((double)tot)/5;
    std::cout << "Time taken for final GPU algo = " << std::fixed << std::setprecision(1) << time << "\n";
    outfile << " " << std::fixed << std::setprecision(1) << time;
    tot = 0;
    for (int i = 0; i < 5; i++) {
      tot = tot + naivesingle<int, int>(output, starter, stopper, 0, 0, size);
    }
    time = ((double)tot)/5;
    std::cout << "Time taken for naive sequential algo on GPU = " << std::fixed << std::setprecision(1) << time << "\n";
    outfile << " " << std::fixed << std::setprecision(1) << time;
    tot = 0;
    for (int i = 0; i < 5; i++) {
      tot = tot + naivemulti<int, int>(output, starter, stopper, 0, 0, size);
    }
    time = ((double)tot)/5;
    std::cout << "Time taken for naive sequential algo parallely on GPU = " << std::fixed << std::setprecision(1) << time << "\n";
    outfile << " " << std::fixed << std::setprecision(1) << time;
    tot = 0;
    for (int i = 0; i < 5; i++) {
      auto start2 = std::chrono::high_resolution_clock::now();
      foo<int, int>(output, starter, stopper, 0, 0, size);
      auto stop2 = std::chrono::high_resolution_clock::now();
      auto time2 = std::chrono::duration_cast<std::chrono::nanoseconds>(stop2 - start2);
      tot += time2.count();
    }
    time = ((double)tot)/5;
    std::cout << "Time taken for CPU = " << std::fixed << std::setprecision(1) << time << "\n";
    outfile << " " << std::fixed << std::setprecision(1) << time << "\n";
  }
  return 0;
}
