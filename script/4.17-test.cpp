#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <stdexcept>

// Function to perform multinomial sampling similar to torch.multinomial(prob, 1)
int multinomial_sample(const std::vector<float>& probabilities) {
    // Check if probabilities are valid
    if (probabilities.empty()) {
        throw std::invalid_argument("Probability vector is empty.");
    }

    // Calculate the sum of probabilities
    float sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0f);
    if (sum <= 0.0f) {
        throw std::invalid_argument("Sum of probabilities must be positive.");
    }

    // Normalize probabilities to sum to 1
    std::vector<float> normalized_probs;
    normalized_probs.reserve(probabilities.size());
    for (const auto& p : probabilities) {
        normalized_probs.push_back(p / sum);
    }

    // Create a discrete distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(normalized_probs.begin(), normalized_probs.end());

    // Sample a single index
    return dist(gen);
}

int main() {
    // Example probability vector
    std::vector<float> prob = {0.1f, 0.3f, 0.4f, 0.2f};

    try {
        // Perform sampling
        int sampled_index = multinomial_sample(prob);
        std::cout << "Sampled Index: " << sampled_index << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}