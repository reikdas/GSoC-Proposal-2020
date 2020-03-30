#include <iostream>
#include <fstream>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
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

// https://stackoverflow.com/a/14038590/4647107
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

template <typename T, typename C>
__global__
void prefix_sum1(T* base, const C* basestart, const C* basestop, int64_t basestartoffset, int64_t basestopoffset, int length, T* sums) {
  int thid = threadIdx.x + (blockIdx.x * blockDim.x);
  extern __shared__ T temp[];
  int pout = 0, pin = 1;
  if (thid < length) {
    temp[threadIdx.x] = basestop[basestopoffset + thid] - basestart[basestartoffset + thid];
    __syncthreads();
    for (int offset = 1; offset < 1024; offset *=2) {
      pout = 1 - pout;
      pin = 1 - pout;
      if (threadIdx.x >= offset) {
        temp[pout*1024 + threadIdx.x] = temp[pin*1024 + threadIdx.x - offset] + temp[pin*1024 + threadIdx.x];
      }
      else {
        temp[pout*1024 + threadIdx.x] = temp[pin*1024 + threadIdx.x];
      }
      __syncthreads();
    }
    base[thid] = temp[pout*1024 + threadIdx.x];
    __syncthreads();
    if ((thid == 1023) || ((blockIdx.x != 0) && (thid == ((1024 * (blockIdx.x + 1))-1))) || (thid == length-1)) {
        sums[blockIdx.x] = base[thid];
    }
  }
}

// Need another kernel because of conditional __syncthreads()
template <typename T>
__global__
void prefix_sum2(T* base, int length) {
  int thid = threadIdx.x + (blockIdx.x * blockDim.x);
  extern __shared__ T temp[];
  int pout = 0, pin = 1;
  if (thid < length) {
    temp[thid] = base[thid];
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
    base[thid] = temp[pout*length + thid];
  }
}

template<typename T>
__global__
void adder(T* base, T* sums, int64_t length) {
  int thid = threadIdx.x + (blockIdx.x * blockDim.x);
  if (blockIdx.x != 0 && thid < length)
    base[thid] += sums[blockIdx.x - 1];
}

template <typename T, typename C>
int offload(T* base, C* basestart1, C* basestop1, int64_t basestartoffset, int64_t basestopoffset, int64_t length) {
  int block, thread=1024;
  if (length > 1024) {
    if (length%1024 != 0)
      block = (length / 1024) + 1;
    else
      block = length/1024;
  }
  else {
    block = 1;
  }
  int modlength = block*thread;
  // Padding the input arrays
  C basestart[modlength], basestop[modlength];
  for (int i=0; i<modlength; i++) {
    if (i<length){
      basestart[i] = basestart1[i];
      basestop[i] = basestop1[i];
    }
    else {
      basestart[i] = 0;
      basestop[i] = 0;
    }
  }
  T* d_tooffsets, * d_sums;
  C* d_fromstarts, * d_fromstops;
  gpuErrchk(cudaMalloc((void**)&d_tooffsets, (modlength+1) * sizeof(T)));
  gpuErrchk(cudaMalloc((void**)&d_fromstarts, modlength * sizeof(C)));
  gpuErrchk(cudaMemcpy(d_fromstarts, basestart, modlength * sizeof(C), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc((void**)&d_fromstops, modlength * sizeof(C)));
  gpuErrchk(cudaMemcpy(d_fromstops, basestop, modlength * sizeof(C), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc((void**)&d_sums, block*sizeof(T)));
  auto start1 = std::chrono::high_resolution_clock::now();
  prefix_sum1<T, C><<<block, thread, thread*2*sizeof(T)>>>(d_tooffsets, d_fromstarts, d_fromstops, basestartoffset, basestopoffset, modlength, d_sums);
  cudaDeviceSynchronize();
  prefix_sum2<T><<<1, block, block*2*sizeof(T)>>>(d_sums, block);
  cudaDeviceSynchronize();
  adder<T><<<block, thread>>>(d_tooffsets, d_sums, modlength);
  cudaDeviceSynchronize();
  auto stop1 = std::chrono::high_resolution_clock::now();
  auto time1 = std::chrono::duration_cast<std::chrono::nanoseconds>(stop1 - start1);
  gpuErrchk(cudaMemcpy(base, d_tooffsets, (length + 1) * sizeof(T), cudaMemcpyDeviceToHost));
  base[length] = base[length - 1] + basestop[length - 1 + basestopoffset] - basestart[length - 1 + basestartoffset];
  gpuErrchk(cudaFree(d_tooffsets));
  gpuErrchk(cudaFree(d_fromstarts));
  gpuErrchk(cudaFree(d_fromstops));
  gpuErrchk(cudaFree(d_sums));
  auto time = time1.count();
  return (int)time;
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
  for (int counter=10; counter<400000; counter+=1000) {
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
    std::cout << "Time taken for final GPU Thrust algo = " << std::fixed << std::setprecision(1) << time << "\n";
    outfile << " " << std::fixed << std::setprecision(1) << time;
    tot = 0;
    for (int i=0; i<5; i++) {
      tot = tot + offload<int, int>(output, starter, stopper, 0, 0, size);
    }
    time = ((double)tot)/5;
    std::cout << "Time taken for final Hillis Steele algo = " << std::fixed << std::setprecision(1) << time << "\n";
    outfile << " " << std::fixed << std::setprecision(1) << time << "\n";
  }
  return 0;
}
