#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>

template <typename C, typename T>
__global__
void awkward_listarray_compact_offsets(T* tooffsets, const C* fromstarts, const C* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length) {
	tooffsets[0] = 0;
	int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	for (int i = 0; i < length; i++) {
		if (i == idx) {
			C start = fromstarts[startsoffset + i];
			C stop = fromstops[stopsoffset + i];
			assert(start < stop);
			tooffsets[i + 1] = tooffsets[i] + (stop - start);
		}
		__syncthreads();
	}
}

int main() {
	int tooffsets[6];
	int fromstarts[] = { 11, 12, 13, 14, 15 };
	int fromstops[] = { 21, 22, 23, 24, 25 };
	int* d_tooffsets, * d_fromstarts, * d_fromstops;
	cudaMalloc((void**)&d_tooffsets, 6 * sizeof(int));
	//cudaMemcpy(d_tooffsets, tooffsets, 6 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_fromstarts, 5 * sizeof(int));
	cudaMemcpy(d_fromstarts, fromstarts, 5 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_fromstops, 5 * sizeof(int));
	cudaMemcpy(d_fromstops, fromstops, 5 * sizeof(int), cudaMemcpyHostToDevice);
	awkward_listarray_compact_offsets <int, int> << <1, 5 >> > (d_tooffsets, d_fromstarts, d_fromstops, 0, 0, 5);
	//cudaDeviceSynchronize();
	cudaMemcpy(tooffsets, d_tooffsets, 6 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_tooffsets);
	cudaFree(d_fromstarts);
	cudaFree(d_fromstops);
	for (int i = 0; i < 6; i++) {
		std::cout << tooffsets[i] << "\n";
	}
	return 0;
}