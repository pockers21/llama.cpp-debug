#include <algorithm>
#include <vector>
#include <iostream>

// Function to perform searchsorted equivalent using std::lower_bound
std::vector<size_t> searchsorted(const std::vector<int>& sorted_vec, const std::vector<int>& query, bool right = false) {
    std::vector<size_t> indices;
    indices.reserve(query.size());

    for (const auto& val : query) {
        // Find the insertion point
        auto it = right ? std::upper_bound(sorted_vec.begin(), sorted_vec.end(), val)
                       : std::lower_bound(sorted_vec.begin(), sorted_vec.end(), val);
        // Calculate the index
        size_t index = std::distance(sorted_vec.begin(), it);
        indices.push_back(index);
    }

    return indices;
}

int main() {
    // Example sorted vector
    std::vector<int> top_scores_index = {1, 3, 5, 7, 9};

    // Example draft_parents vector
    std::vector<int> draft_parents = {2, 4, 6};

    // Adjust draft_parents by subtracting 1
    std::vector<int> adjusted_parents;
    adjusted_parents.reserve(draft_parents.size());
    for (const auto& val : draft_parents) {
        adjusted_parents.push_back(val - 1);
    }

    // Perform searchsorted with right=false
    std::vector<size_t> mask_index = searchsorted(top_scores_index, adjusted_parents, false);

    // Output the results
    std::cout << "mask_index: ";
    for (const auto& idx : mask_index) {
        std::cout << idx << " ";
    }
    std::cout << std::endl;

    return 0;
}