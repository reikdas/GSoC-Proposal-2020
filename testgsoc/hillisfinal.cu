#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>

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
    if (thid == 0) {
      temp[threadIdx.x] = 0;
    }
    else {
      temp[threadIdx.x] = basestop[basestopoffset + thid - 1] - basestart[basestartoffset + thid - 1];
    }
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
void offload(T* base, C* basestart1, C* basestop1, int64_t basestartoffset, int64_t basestopoffset, int64_t length) {
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
  prefix_sum1<T, C><<<block, thread, thread*2*sizeof(T)>>>(d_tooffsets, d_fromstarts, d_fromstops, basestartoffset, basestopoffset, modlength, d_sums);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  prefix_sum2<T><<<1, block, block*2*sizeof(T)>>>(d_sums, block);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  adder<T><<<block, thread>>>(d_tooffsets, d_sums, modlength);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(base, d_tooffsets, (length + 1) * sizeof(T), cudaMemcpyDeviceToHost));
  base[length] = base[length - 1] + basestop[length - 1 + basestopoffset] - basestart[length - 1 + basestartoffset];
  gpuErrchk(cudaFree(d_tooffsets));
  gpuErrchk(cudaFree(d_fromstarts));
  gpuErrchk(cudaFree(d_fromstops));
  gpuErrchk(cudaFree(d_sums));
}

int main() {
  const int size = 400000;
  int base[size + 1], basestart[size], basestop[size];
  for (int i = 0; i < size; i++) {
    basestart[i] = i;
    basestop[i] = i + 10;
  }
  offload<int, int>(base, basestart, basestop, 0, 0, size);
  for (int i = 0; i < size + 1; i++) {
      std::cout << base[i] << "\n";
  }
  return 0;
}
