#include <vector>
#include <cstdint>

std::vector<float> repeat_tensor(const std::vector<float>& last_hidden, int64_t bs, int64_t hidden_dim, int64_t top_k) {
    std::vector<float> result(bs * top_k * hidden_dim);

    for (int64_t i = 0; i < bs; ++i) {
        for (int64_t j = 0; j < top_k; ++j) {
            for (int64_t k = 0; k < hidden_dim; ++k) {
                result[(i * top_k + j) * hidden_dim + k] = last_hidden[i * hidden_dim + k];
            }
        }
    }

    return result;
}

int main() {
    int64_t bs = 2;
    int64_t hidden_dim = 3;
    int64_t top_k = 4;

    std::vector<float> last_hidden = {1, 2, 3, 4, 5, 6}; // (2, 3) 张量

    std::vector<float> repeated = repeat_tensor(last_hidden, bs, hidden_dim, top_k);

    for (int64_t i = 0; i < bs * top_k * hidden_dim; ++i) {
        printf("%.1f ", repeated[i]);
        if ((i + 1) % (hidden_dim * top_k) == 0) printf("\n");
    }

    return 0;
}