#include <algorithm>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> numbers = {5, 2, 9, 1, 5, 6};

    std::sort(numbers.begin(), numbers.end(), [](int a, int b) {
        return a > b; 
    });

    std::cout << "Sorted numbers (descending): ";
    for(const auto& num : numbers) {
        std::cout << num << " ";
    }
    return 0;
}