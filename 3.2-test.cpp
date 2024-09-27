#include  <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

struct llama_token_data {
    int32_t id;
    float logit;
    float p;
};

struct llama_token_data_array {
    llama_token_data * data;
    size_t size;
    bool sorted;
};

void llama_sample_top_k_impl(llama_token_data_array * candidates, int k, size_t min_keep) {
    k = std::min(k, (int) candidates->size);
    k = std::max(k, (int) min_keep);

    std::partial_sort(
        candidates->data,
        candidates->data + k,
        candidates->data + candidates->size,
        [](const llama_token_data & a, const llama_token_data & b) {
            return a.logit > b.logit;
        }
    );

    candidates->size = k;
}

std::vector<std::vector<llama_token_data>> batch_top_k(const std::vector<std::vector<float>>& last_p, int top_k) {
    int bs = last_p.size();
    int vocab_size = last_p[0].size();

    std::vector<std::vector<llama_token_data>> results(bs);

    for (int i = 0; i < bs; ++i) {
        std::vector<llama_token_data> candidates(vocab_size);
        for (int j = 0; j < vocab_size; ++j) {
            candidates[j].id = j;
            candidates[j].logit = std::log(last_p[i][j]);
            candidates[j].p = last_p[i][j];
        }

        llama_token_data_array candidates_array = { candidates.data(), vocab_size, false };
        llama_sample_top_k_impl(&candidates_array, top_k, top_k);

        results[i] = std::vector<llama_token_data>(candidates_array.data, candidates_array.data + candidates_array.size);
    }

    return results;
}

int main() {
    // 模拟批次数据
    std::vector<std::vector<float>> last_p = {
        {0.1, 0.05, 0.2, 0.15, 0.3, 0.05, 0.02, 0.03, 0.08, 0.02},
        {0.05, 0.1, 0.15, 0.3, 0.05, 0.2, 0.05, 0.02, 0.03, 0.05}
    };
    int top_k = 3;

    auto results = batch_top_k(last_p, top_k);

    // 输出结果
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "Sample " << i << " top " << top_k << " values and indices:\n";
        for (const auto& token : results[i]) {
            std::cout << "Value: " << token.p << ", Index: " << token.id << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}

