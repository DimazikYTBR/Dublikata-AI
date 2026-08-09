#include <string>
#include <vector>
#include <algorithm>

class AlphabetManager {
private:
    std::string alphabet;

public:
    AlphabetManager() {
        // Русский, английский, цифры, знаки и перенос строки
        alphabet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZабвгдежзийклмнопрстуфхцчшщъыьэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789.,!?-_\n";
    }

    int size() const {
        return static_cast<int>(alphabet.size());
    }

    int encode(char c) const {
        size_t found = alphabet.find(c);
        if (found != std::string::npos) {
            return static_cast<int>(found);
        }
        return 0; // Пробел по умолчанию для неизвестных символов
    }

    char decode(int token) const {
        if (token >= 0 && token < static_cast<int>(alphabet.size())) {
            return alphabet[token];
        }
        return ' ';
    }
};
