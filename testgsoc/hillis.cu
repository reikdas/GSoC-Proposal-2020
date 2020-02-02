#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>

template <typename C, typename T>
__global__
void awkward_listarray_compact_offsets(T* tooffsets, const C* fromstarts, const C* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length) {
	extern __shared__ T temp[]; // allocated on invocation
	int thid = threadIdx.x;
	int pout = 0, pin = 1;
	// load input into shared memory.
	// Exclusive scan: shift right by one and set first element to 0
	temp[thid] = (thid > 0) ? (fromstops[thid - 1]-fromstarts[thid-1]) : 0;
	__syncthreads();
	for (int offset = 1; offset < length; offset <<= 1)
	{
		pout = 1 - pout; // swap double buffer indices
		pin = 1 - pout;
		if (thid >= offset)
			temp[pout * length + thid] += temp[pin * length + thid - offset];
		else
			temp[pout * length + thid] = temp[pin * length + thid];
		__syncthreads();
	}
	tooffsets[thid] = temp[pout * length + thid]; // write output
}

int main() {
	int tooffsets[6];
	int fromstarts[] = { 11, 12, 13, 14, 15 };
	int fromstops[] = { 21, 22, 23, 24, 25 };
	int* d_tooffsets, * d_fromstarts, * d_fromstops;
	cudaMalloc((void**)&d_tooffsets, 6 * sizeof(int));
	cudaMalloc((void**)&d_fromstarts, 5 * sizeof(int));
	cudaMemcpy(d_fromstarts, fromstarts, 5 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_fromstops, 5 * sizeof(int));
	cudaMemcpy(d_fromstops, fromstops, 5 * sizeof(int), cudaMemcpyHostToDevice);
	awkward_listarray_compact_offsets <int, int> << <1, 5 >> > (d_tooffsets, d_fromstarts, d_fromstops, 0, 0, 5);
	cudaMemcpy(tooffsets, d_tooffsets, 6 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_tooffsets);
	cudaFree(d_fromstarts);
	cudaFree(d_fromstops);
	for (int i = 0; i < 6; i++) {
		std::cout << tooffsets[i] << "\n";
	}
	return 0;
}