#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>

struct TinyTextGenModel {
    int vocab_size = 256;
    int hidden_size = 64; // будет пересчитан под размер файла

    std::vector<float> w_embed;
    std::vector<float> w_hidden;
    std::vector<float> w_out;

    // Распаковка 5-битных значений из потока байт в float [-1, 1].
    // Предполагает, что веса упакованы подряд по 5 бит без выравнивания
    // по байтам (самая простая/общая схема, без group-scale).
    static std::vector<float> unpack_int5(const std::vector<uint8_t>& raw, size_t count) {
        std::vector<float> out;
        out.reserve(count);
        size_t bit_pos = 0;
        for (size_t i = 0; i < count; ++i) {
            size_t byte_idx = bit_pos / 8;
            size_t bit_off  = bit_pos % 8;
            if (byte_idx + 1 >= raw.size()) break; // не выходим за пределы буфера
            uint16_t chunk = uint16_t(raw[byte_idx]) | (uint16_t(raw[byte_idx + 1]) << 8);
            uint16_t val5 = (chunk >> bit_off) & 0x1F; // 5 бит -> значение 0..31
            out.push_back((float(val5) / 31.0f) * 2.0f - 1.0f); // нормализация в [-1,1]
            bit_pos += 5;
        }
        return out;
    }

    // Подбирает hidden_size так, чтобы суммарное число весов сети
    // (hidden^2 + 2*vocab*hidden) максимально близко соответствовало
    // количеству значений, реально доступных в файле.
    void fit_architecture_to_param_count(size_t available_params) {
        double a = 1.0, b = 2.0 * vocab_size, c = -double(available_params);
        double h = (-b + std::sqrt(b * b - 4 * a * c)) / (2 * a);
        hidden_size = std::max(1, (int)std::floor(h));
    }

    void allocate() {
        w_embed.assign((size_t)vocab_size * hidden_size, 0.0f);
        w_hidden.assign((size_t)hidden_size * hidden_size, 0.0f);
        w_out.assign((size_t)hidden_size * vocab_size, 0.0f);
    }

    bool load_weights_int5(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) return false;

        std::streamsize file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        if (file_size <= 0) return false;

        std::vector<uint8_t> raw((size_t)file_size);
        file.read(reinterpret_cast<char*>(raw.data()), file_size);
        if (file.gcount() != file_size) {
            std::cerr << "[!] Прочитано только " << file.gcount()
                      << " из " << file_size << " байт\n";
        }
        file.close();

        // Сколько 5-битных значений реально можно извлечь из файла
        size_t max_int5_values = (raw.size() * 8) / 5;

        // Подгоняем архитектуру под фактический размер файла
        fit_architecture_to_param_count(max_int5_values);
        allocate();

        size_t needed = w_embed.size() + w_hidden.size() + w_out.size();
        size_t to_read = std::min(needed, max_int5_values);

        std::vector<float> flat = unpack_int5(raw, to_read);

        std::cout << "[i] Файл: " << file_size << " байт, доступно INT5-значений: "
                  << max_int5_values << "\n"
                  << "[i] Подобран hidden_size=" << hidden_size
                  << " (нужно весов: " << needed << ", прочитано: " << flat.size() << ")\n";

        size_t idx = 0;
        for (size_t i = 0; i < w_embed.size()  && idx < flat.size(); ++i, ++idx) w_embed[i]  = flat[idx];
        for (size_t i = 0; i < w_hidden.size() && idx < flat.size(); ++i, ++idx) w_hidden[i] = flat[idx];
        for (size_t i = 0; i < w_out.size()    && idx < flat.size(); ++i, ++idx) w_out[i]    = flat[idx];

        if (idx < needed) {
            std::cerr << "[!] Файла хватило только на " << idx << " из " << needed
                      << " весов (" << (100.0 * idx / needed) << "%). Остаток заполнен нулями.\n";
        }

        return idx > 0; // не падаем, даже если хватило только на часть сети
    }

    int forward(int input_token, const std::vector<float>& context_vector) {
        std::vector<float> hidden(hidden_size, 0.0f);
        for (int i = 0; i < hidden_size; ++i) {
            hidden[i] = w_embed[(size_t)input_token * hidden_size + i];
            if (!context_vector.empty()) hidden[i] += context_vector[i % context_vector.size()];
        }

        std::vector<float> next_hidden(hidden_size, 0.0f);
        for (int i = 0; i < hidden_size; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < hidden_size; ++j) sum += hidden[j] * w_hidden[(size_t)j * hidden_size + i];
            next_hidden[i] = std::max(0.0f, sum);
        }

        std::vector<float> logits(vocab_size, 0.0f);
        float max_logit = -1e9f;
        for (int i = 0; i < vocab_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) logits[i] += next_hidden[j] * w_out[(size_t)j * vocab_size + i];
            if (logits[i] > max_logit) max_logit = logits[i];
        }

        float sum_exp = 0.0f;
        std::vector<float> probs(vocab_size);
        for (int i = 0; i < vocab_size; ++i) { probs[i] = std::exp(logits[i] - max_logit); sum_exp += probs[i]; }

        float r = static_cast<float>(rand()) / RAND_MAX * sum_exp;
        float cumul = 0.0f;
        for (int i = 0; i < vocab_size; ++i) { cumul += probs[i]; if (cumul >= r) return i; }
        return 0;
    }
};

int main() {
    std::srand(42);
    TinyTextGenModel model;
    std::string weights_file = "weights.bin";

    if (!model.load_weights_int5(weights_file)) {
        std::cerr << "[!] Не удалось прочитать веса из '" << weights_file << "'\n";
        return 1;
    }

    int seed_token = 'H';
    int max_length = 100;
    std::vector<float> context = {0.1f, 0.5f, 0.2f};

    std::cout << "--- НАЧАЛО ГЕНЕРАЦИИ ---\n";
    int current_token = seed_token;
    for (int step = 0; step < max_length; ++step) {
        current_token = model.forward(current_token, context);
        if (current_token >= 32 && current_token <= 126) std::cout << static_cast<char>(current_token);
        else if (current_token == 10) std::cout << "\n";
    }
    std::cout << "\n--- КОНЕЦ ГЕНЕРАЦИИ ---\n";
    return 0;
}
