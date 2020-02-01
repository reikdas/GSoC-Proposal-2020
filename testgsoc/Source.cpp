#include <iostream>

template <typename C, typename T>
void awkward_listarray_compact_offsets(T* tooffsets, const C* fromstarts, const C* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length) {
	tooffsets[0] = 0;
	for (int64_t i = 0; i < length; i++) {
		C start = fromstarts[startsoffset + i];
		C stop = fromstops[stopsoffset + i];
		if (stop < start) {
		}
		tooffsets[i + 1] = tooffsets[i] + (stop - start);
	}
}

int test_main() {
	int tooffsets[6];
	int fromstarts[] = { 11, 12, 13, 14, 15 };
	int fromstops[] = { 21, 22, 23, 24, 25 };
	awkward_listarray_compact_offsets<int, int>(tooffsets, fromstarts, fromstops, 0, 0, 5);
	for (int i = 0; i < 6; i++) {
		std::cout << tooffsets[i] << "\n";
	}
	return 1;
}