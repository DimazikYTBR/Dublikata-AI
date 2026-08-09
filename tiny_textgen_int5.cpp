#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <algorithm>

struct TinyTextGenModel {
    std::string alphabet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZабвгдежзийклмнопрстуфхцчшщъыьэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789.,!?-_";
    int vocab_size = alphabet.size();
    int hidden_size = 32;

    std::vector<float> w_embed;
    std::vector<float> w_hidden;
    std::vector<float> w_out;

    int char_to_token(char c) {
        size_t found = alphabet.find(c);
        if (found != std::string::npos) return static_cast<int>(found);
        return 0; 
    }

    char token_to_char(int token) {
        if (token >= 0 && token < static_cast<int>(alphabet.size())) {
            return alphabet[token];
        }
        return ' ';
    }

    static std::vector<float> unpack_int5(const std::vector<uint8_t>& raw, size_t count) {
        std::vector<float> out;
        if (raw.empty() || count == 0) return out;
        out.reserve(count);
        size_t bit_pos = 0;
        
        for (size_t i = 0; i < count; ++i) {
            size_t byte_idx = bit_pos / 8;
            size_t bit_off  = bit_pos % 8;
            
            if (byte_idx + 1 >= raw.size()) {
                out.push_back(0.0f);
                bit_pos += 5;
                continue;
            }
            
            uint16_t chunk = uint16_t(raw[byte_idx]) | (uint16_t(raw[byte_idx + 1]) << 8);
            uint16_t val5 = (chunk >> bit_off) & 0x1F; 
            out.push_back((float(val5) / 31.0f) * 2.0f - 1.0f);
            bit_pos += 5;
        }
        return out;
    }

    void allocate() {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.1f);

        w_embed.resize((size_t)vocab_size * hidden_size);
        for (auto& w : w_embed) w = dist(rng);

        w_hidden.resize((size_t)hidden_size * hidden_size);
        for (auto& w : w_hidden) w = dist(rng);

        w_out.resize((size_t)hidden_size * vocab_size);
        for (auto& w : w_out) w = dist(rng);
    }

    bool load_weights_int5(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            allocate();
            return false;
        }

        std::streamsize file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        if (file_size <= 0) {
            file.close();
            allocate();
            return false;
        }

        std::vector<uint8_t> raw((size_t)file_size);
        file.read(reinterpret_cast<char*>(raw.data()), file_size);
        file.close();

        size_t max_int5_values = (raw.size() * 8) / 5;
        allocate();

        size_t needed = w_embed.size() + w_hidden.size() + w_out.size();
        size_t to_read = std::min(needed, max_int5_values);

        std::vector<float> flat = unpack_int5(raw, to_read);

        size_t idx = 0;
        for (size_t i = 0; i < w_embed.size()  && idx < flat.size(); ++i, ++idx) w_embed[i]  = flat[idx];
        for (size_t i = 0; i < w_hidden.size() && idx < flat.size(); ++i, ++idx) w_hidden[i] = flat[idx];
        for (size_t i = 0; i < w_out.size()    && idx < flat.size(); ++i, ++idx) w_out[i]    = flat[idx];

        return true;
    }

    int forward(int input_token, const std::vector<float>& context_vector) {
        std::vector<float> hidden(hidden_size, 0.0f);
        for (int i = 0; i < hidden_size; ++i) {
            size_t idx = (size_t)input_token * hidden_size + i;
            if (idx < w_embed.size()) hidden[i] = w_embed[idx];
            if (!context_vector.empty()) hidden[i] += context_vector[i % context_vector.size()];
        }

        std::vector<float> next_hidden(hidden_size, 0.0f);
        for (int i = 0; i < hidden_size; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < hidden_size; ++j) {
                size_t idx = (size_t)j * hidden_size + i;
                if (idx < w_hidden.size()) sum += hidden[j] * w_hidden[idx];
            }
            next_hidden[i] = std::max(0.0f, sum);
        }

        std::vector<float> logits(vocab_size, 0.0f);
        float max_logit = -1e9f;
        for (int i = 0; i < vocab_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                size_t idx = (size_t)j * vocab_size + i;
                if (idx < w_out.size()) logits[i] += next_hidden[j] * w_out[idx];
            }
            if (logits[i] > max_logit) max_logit = logits[i];
        }

        float temperature = 0.8f;
        float sum_exp = 0.0f;
        std::vector<float> probs(vocab_size, 0.0f);

        for (int i = 0; i < vocab_size; ++i) { 
            probs[i] = std::exp(std::clamp((logits[i] - max_logit) / temperature, -20.0f, 0.0f)); 
            sum_exp += probs[i]; 
        }

        if (sum_exp <= 0.0f) return 0;

        static int last_token = 0;
        static int repeat_count = 0;
        
        float r = static_cast<float>(rand()) / RAND_MAX * sum_exp;
        float cumul = 0.0f;
        int chosen_token = 0;
        
        for (int i = 0; i < vocab_size; ++i) { 
            cumul += probs[i]; 
            if (cumul >= r) { 
                chosen_token = i; 
                break;
            } 
        }

        if (chosen_token == last_token) {
            repeat_count++;
            if (repeat_count > 4) {
                chosen_token = rand() % vocab_size;
                repeat_count = 0;
            }
        } else {
            repeat_count = 0;
            last_token = chosen_token;
        }

        return chosen_token;
    }
};

int main() {
    std::srand(1337);
    TinyTextGenModel model;
    std::string weights_file = "weights.bin";

    model.load_weights_int5(weights_file);

    char seed_char = 'H';
    int current_token = model.char_to_token(seed_char);
    int max_length = 64;
    std::vector<float> context = {0.1f, 0.3f, 0.7f};

    for (int step = 0; step < max_length; ++step) {
        current_token = model.forward(current_token, context);
        std::cout << model.token_to_char(current_token);
    }
    std::cout << std::endl;
    return 0;
}
