#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_set>



std::vector<int> unique_ordered(const std::vector<int>& vec) {
    std::unordered_set<int> seen;
    std::vector<int> unique;
    for (const auto& elem : vec) {
        if (seen.find(elem) == seen.end()) {
            seen.insert(elem);
            unique.push_back(elem);
        }
    }
    return unique;
}

int main() {
    std::vector<int> mask_index = {3, 1, 2, 3, 4, 2, 5, 1};

    std::vector<int> unique_ordered_vec = unique_ordered(mask_index);
    std::cout << "Unique (ordered): ";
    for (const auto& num : unique_ordered_vec) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}