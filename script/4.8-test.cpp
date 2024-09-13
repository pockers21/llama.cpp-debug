#include <vector>
#include <cstdint>

std::vector<float> create_eye_tensor(int64_t top_k) {
    std::vector<float> result(top_k * top_k, 0.0f);

    for (int64_t i = 0; i < top_k; ++i) {
        result[i * top_k + i] = 1.0f;
    }

    return result;
}

int main() {
    int64_t top_k = 3;
    std::vector<float> eye_tensor = create_eye_tensor(top_k);

    for (int64_t i = 0; i < top_k; ++i) {
        for (int64_t j = 0; j < top_k; ++j) {
            printf("%.1f ", eye_tensor[i * top_k + j]);
        }
        printf("\n");
    }

    return 0;
}