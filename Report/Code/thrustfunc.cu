#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include "helper_cuda.h"

__device__
void success_cuda(Error* err) {
  err.str = nullptr;
  err.identity = kSliceNone;
  err.attempt = kSliceNone;
  err.extra = 0;
}

__device__
void failure_cuda(const char* str, int64_t identity, int64_t attempt, Error* err) {
  err.str = str;
  err.identity = identity;
  err.attempt = attempt;
  err.extra = 0;
}

template <typename T, typename C>
__global__
void sub(T* output, const C* starter, const C* stopper, int64_t startsoffset, int64_t stopsoffset, int64_t n, Error* err) {
  __shared__ int flag[1];
  int thid = threadIdx.x + blockIdx.x * blockDim.x;
  if (thid == 0) flag[0] = 0;
  if (thid < n) {
    C start = starter[thid + startsoffset];
    C stop = stopper[thid + stopsoffset];
    if (stop < start) {
      failure_cuda("stops[i] < starts[i]", thid , kSliceNone, err);
      flag[0] = 1;
    }
    output[thid] = stop - start;
  }
  __syncthreads();
  if (flag[0] != 1) {
    success_cuda(err);
  }
}

template <typename T, typename C>
void prefix_sum(T* output, const C* arr, const C* arr2, int64_t startsoffset, int64_t stopsoffset, int64_t length, Error* err) {
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
  checkCudaErrors(cudaMalloc((void**)&d_output, length * sizeof(T)));
  checkCudaErrors(cudaMalloc((void**)&d_arr, length * sizeof(C)));
  checkCudaErrors(cudaMemcpy(d_arr, arr, length * sizeof(C), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void**)&d_arr2, length * sizeof(C)));
  checkCudaErrors(cudaMemcpy(d_arr2, arr2, length * sizeof(C), cudaMemcpyHostToDevice));
  sub<T, C><<<block, thread>>>(d_output, d_arr, d_arr2, startsoffset, stopsoffset, length, err);
  checkCudaErrors(cudaDeviceSynchronize());
  thrust::device_vector<T> data(d_output, d_output+length);
  thrust::device_vector<T> temp(data.size() + 1);
  thrust::exclusive_scan(data.begin(), data.end(), temp.begin());
  temp[data.size()] = data.back() + temp[data.size() - 1];
  thrust::copy(temp.begin(), temp.end(), output);
  checkCudaErrors(cudaFree(d_output));
  checkCudaErrors(cudaFree(d_arr));
  checkCudaErrors(cudaFree(d_arr2));
}
