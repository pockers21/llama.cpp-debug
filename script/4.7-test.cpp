#include <vector>
#include <cstdint>

std::vector<int32_t> manual_arange(int32_t top_k) {
    std::vector<int32_t> result(top_k);
    for (int32_t i = 0; i < top_k; ++i) {
        result[i] = i;
    }
    return result;
}

int main() {
    int32_t top_k = 5;
    std::vector<int32_t> arange_result = manual_arange(top_k);

    for (int32_t val : arange_result) {
        printf("%d ", val);
    }
    printf("\n");

    return 0;
}