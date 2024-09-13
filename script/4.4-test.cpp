#include <iostream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <numeric>

std::vector<float> concatenate_and_flatten(const std::vector<std::vector<float>>& scores_list) {
    size_t total_elements = std::accumulate(scores_list.begin(), scores_list.end(), 0,
        [](size_t sum, const std::vector<float>& v) { return sum + v.size(); });

    std::vector<float> result;
    result.reserve(total_elements);

    for (const auto& scores : scores_list) {
        result.insert(result.end(), scores.begin(), scores.end());
    }

    return result;
}

int main() {
    std::vector<std::vector<float>> scores_list = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    std::vector<float> flattened_scores = concatenate_and_flatten(scores_list);

    for (float score : flattened_scores) {
        std::cout<< score << std::endl;;
    }
    printf("\n");

    return 0;
}