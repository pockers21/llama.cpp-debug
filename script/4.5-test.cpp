#include <vector>
#include <cstdint>

std::vector<int64_t> create_zero_tensor() {
    return std::vector<int64_t>(1, 0);
}

int main() {
    std::vector<int64_t> zero_tensor = create_zero_tensor();

    if (zero_tensor.size() == 1 && zero_tensor[0] == 0) {
        printf("Successfully created a zero tensor\n");
    } else {
        printf("Failed to create a zero tensor\n");
    }

    return 0;
}