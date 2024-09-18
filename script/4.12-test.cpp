#include <iostream>
#include <vector>
#include <limits>

// Function to expand and invert the attention mask
std::vector<std::vector<std::vector<std::vector<float>>>> expand_mask(
    const std::vector<std::vector<float>>& mask,
    float dtype = 1.0f,
    int tgt_len = -1) {


    // Get batch size and source length
    int bsz = mask.size();
    int src_len = mask.empty() ? 0 : mask[0].size();

    // Determine target length
    if (tgt_len == -1) {
        tgt_len = src_len;
    }

    // Initialize expanded mask with dimensions [bsz, 1, tgt_len, src_len]
    std::vector<std::vector<std::vector<std::vector<float>>>> expanded_mask(
        bsz, std::vector<std::vector<std::vector<float>>>(
                1, std::vector<std::vector<float>>(
                       tgt_len, std::vector<float>(src_len, 0.0f))));

    // Expand the mask
    for (int i = 0; i < bsz; ++i) {
        for (int t = 0; t < tgt_len; ++t) {
            for (int s = 0; s < src_len; ++s) {
                expanded_mask[i][0][t][s] = mask[i][s] * dtype;
            }
        }
    }

    // Invert the mask: 1.0 - expanded_mask
    std::vector<std::vector<std::vector<std::vector<float>>>> inverted_mask = expanded_mask;
    for (int i = 0; i < bsz; ++i) {
        for (int t = 0; t < tgt_len; ++t) {
            for (int s = 0; s < src_len; ++s) {
                inverted_mask[i][0][t][s] = 1.0f - expanded_mask[i][0][t][s];
            }
        }
    }

    // Logging
    std::cout << "tgt_len: " << tgt_len << ", src_len: " << src_len << std::endl;
    std::cout << "Shape of inverted_mask: [" << inverted_mask.size() << ", "
              << inverted_mask[0].size() << ", " << inverted_mask[0][0].size()
              << ", " << inverted_mask[0][0][0].size() << "]" << std::endl;

    // Replace masked positions with the minimum float value
    for (int i = 0; i < bsz; ++i) {
        for (int t = 0; t < tgt_len; ++t) {
            for (int s = 0; s < src_len; ++s) {
                if (inverted_mask[i][0][t][s] > 0.0f) { // Assuming mask == 1.0f
                    inverted_mask[i][0][t][s] = std::numeric_limits<float>::lowest();
                }
            }
        }
    }

    // Logging the shape after masking
    std::cout << "Shape of ret: [" << inverted_mask.size() << ", "
              << inverted_mask[0].size() << ", " << inverted_mask[0][0].size()
              << ", " << inverted_mask[0][0][0].size() << "]" << std::endl;

    return inverted_mask;

}

int main() {
    // Example usage
    // Create a sample mask tensor of shape [2, 4]
    std::vector<std::vector<float>> mask = {
        {1.0f, 1.0f, 1.0f, 0.0f},
        {1.0f, 1.0f, 0.0f, 0.0f}
    };

    // Specify the target length
    int tgt_len = 5;

    // Call the expand_mask function
    auto expanded = expand_mask(mask, 1.0f, tgt_len);

    // Print the resulting tensor
    std::cout << "Expanded Mask:" << std::endl;
    for (const auto& batch : expanded) {
        for (const auto& channel : batch) {
            for (const auto& tgt : channel) {
                for (const auto& val : tgt) {
                    if (val == std::numeric_limits<float>::lowest()) {
                        std::cout << "-inf ";
                    } else {
                        std::cout << val << " ";
                    }
                }
                std::cout << std::endl;
            }
            std::cout << "-----" << std::endl;
        }
    }

    return 0;
}