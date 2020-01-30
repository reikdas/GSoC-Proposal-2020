#include <iostream>

template <typename C, typename T>
void awkward_indexedarray_flatten_nextcarry(T* tocarry, const C* fromindex, int64_t indexoffset, int64_t lenindex, int64_t lencontent) {
    int64_t k = 0;
    for (int64_t i = 0; i < lenindex; i++) {
        C j = fromindex[indexoffset + i];
        if (j >= lencontent) {
            std::cout << "Index out of range" << "\n";
            exit(0);
        }
        else if (j>= 0) {
            tocarry[k] = j;
            k++;
        }
    }
    std::cout << "Successful!" << "\n";
}

int main() {
    int tocarry[5];
    int fromindex[] = {11, 12, 13, 14, 15};
    awkward_indexedarray_flatten_nextcarry<int, int>(tocarry, fromindex, 0, 5, 20);
    for (int i=0; i<5; i++) {
        std::cout << tocarry[i] << "\n";
    }
    return 1;
}

