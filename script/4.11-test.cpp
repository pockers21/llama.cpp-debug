#include <vector>
#include <cstdint>
#include <limits>
#include <algorithm>

std::vector<float> make_causal_mask(int64_t bsz, int64_t tgt_len, int64_t past_key_values_length = 0) {
    int64_t total_len = tgt_len + past_key_values_length;
    std::vector<float> mask(bsz * total_len * total_len, -std::numeric_limits<float>::infinity());

    for (int64_t b = 0; b < bsz; ++b) {
        for (int64_t i = 0; i < total_len; ++i) {
            for (int64_t j = 0; j <= i; ++j) {
                mask[b * total_len * total_len + i * total_len + j] = 0.0f;
            }
        }
    }

    return mask;
}
int main() {
    int64_t bsz = 1;
    int64_t tgt_len = 4;
    int64_t past_key_values_length = 2;

    std::vector<float> causal_mask = make_causal_mask(bsz, tgt_len, past_key_values_length);

    for (int64_t i = 0; i < tgt_len + past_key_values_length; ++i) {
        for (int64_t j = 0; j < tgt_len + past_key_values_length; ++j) {
            float value = causal_mask[i * (tgt_len + past_key_values_length) + j];
            if (value == -std::numeric_limits<float>::infinity()) {
                printf("-inf ");
            } else {
                printf("%.1f  ", value);
            }
        }
        printf("\n");
    }

    return 0;
}