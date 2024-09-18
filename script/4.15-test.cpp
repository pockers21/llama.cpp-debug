#include <iostream>
#include <vector>

// Function to sum along a specific dimension (dim = 1)
std::vector<std::vector<float>> sum_dim1(const std::vector<std::vector<std::vector<std::vector<float>>>>& tree_mask) {
    std::vector<std::vector<float>> tree_position_ids;

    for (const auto& batch : tree_mask) {
        for (const auto& channel : batch) {
            std::vector<float> summed;
            for (const auto& row : channel) {
                float row_sum = 0.0f;
                for (const auto& val : row) {
                    row_sum += val;
                }
                summed.push_back(row_sum);
            }
            tree_position_ids.push_back(summed);
        }
    }

    return tree_position_ids;
}

int main() {
    std::vector<std::vector<std::vector<std::vector<float>>>> tree_mask = {
        {
            {
                {1.0f, 2.0f, 3.0f, 4.0f},
                {5.0f, 6.0f, 7.0f, 8.0f},
                {9.0f, 10.0f, 11.0f, 12.0f}
            }
        }
    };

    std::vector<std::vector<float>> tree_position_ids = sum_dim1(tree_mask);

    for (const auto& batch : tree_position_ids) {
        for (const auto& sum : batch) {
            std::cout << sum << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}