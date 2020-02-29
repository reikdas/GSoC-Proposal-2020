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

template <typename C, typename T>
__global__
void awkward_listarray_compact_offsets(T* tooffsets, const C* fromstarts, const C* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length, Error* err) {
  __shared__ int flag[1];
  int idx = threadIdx.x + (blockIdx.x * blockDim.x);
  if (idx == 0) {
    tooffsets[0] = 0;
    flag[0] = 0;
  }
  if (idx < length) {
    if (idx == 0) {
      for (int i = 0; i < length; i++) {
        C start = fromstarts[startsoffset + i];
        C stop = fromstops[stopsoffset + i];
        if (stop < start) {
          failure_cuda("stops[i] < starts[i]", i, kSliceNone, err);
          flag[0] = 1;
        }
        tooffsets[i + 1] = tooffsets[i] + (stop - start);
      }
    }
  }
  __syncthreads();
  if (flag[0] != 1) {
    success_cuda(err);
  }
}

