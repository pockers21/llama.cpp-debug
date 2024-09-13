#include <iostream>
#include <vector>
#include <cstring>

struct HiddenState {
    std::vector<float> data;
    int batch_size;
    int seq_length;
    int hidden_size;

    HiddenState(int b, int s, int h) : batch_size(b), seq_length(s), hidden_size(h) {
        data.resize(b * s * h);
    }

    void fill_random() {
        for (auto& val : data) {
            val = static_cast<float>(rand()) / RAND_MAX;
        }
    }
};

std::vector<float> get_last_hidden_state(const HiddenState& hidden_state) {
    std::vector<float> last_hidden(hidden_state.batch_size * hidden_state.hidden_size);

    for (int i = 0; i < hidden_state.batch_size; i++) {
        int offset = i * hidden_state.seq_length * hidden_state.hidden_size + 
                     (hidden_state.seq_length - 1) * hidden_state.hidden_size;
        
        std::memcpy(last_hidden.data() + i * hidden_state.hidden_size, 
                    hidden_state.data.data() + offset, 
                    hidden_state.hidden_size * sizeof(float));
    }

    return last_hidden;
}

int main() {
    HiddenState hidden_state(2, 5, 3);  // batch_size=2, seq_length=5, hidden_size=3
    hidden_state.fill_random();

    std::vector<float> last_hidden = get_last_hidden_state(hidden_state);

    std::cout << "Last hidden state:" << std::endl;
    for (int i = 0; i < hidden_state.batch_size; i++) {
        std::cout << "Batch " << i << ": ";
        for (int j = 0; j < hidden_state.hidden_size; j++) {
            std::cout << last_hidden[i * hidden_state.hidden_size + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
