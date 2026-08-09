#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

class AlphabetManager {
private:
    // Полный набор символов: английский, русский, цифры и знаки препинания
    std::string alphabet;

public:
    AlphabetManager() {
        // Собираем всё в одну строку-алфавит
        alphabet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZабвгдежзийклмнопрстуфхцчшщъыьэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789.,!?-_\n";
    }

    // Получить размер словаря (vocab_size)
    int size() const {
        return static_cast<int>(alphabet.size());
    }

    // Перевод символа в индекс (токен)
    int encode(char c) const {
        size_t found = alphabet.find(c);
        if (found != std::string::npos) {
            return static_cast<int>(found);
        }
        return 0; // Если символ не найден, возвращаем пробел
    }

    // Перевод индекса обратно в символ
    char decode(int token) const {
        if (token >= 0 && token < static_cast<int>(alphabet.size())) {
            return alphabet[token];
        }
        return ' ';
    }

    // Функция для отладки: показать, как строка превращается в массив токенов
    std::vector<int> text_to_tokens(const std::string& text) const {
        std::vector<int> tokens;
        for (char c : text) {
            tokens.push_back(encode(c));
        }
        return tokens;
    }

    // Обратно из токенов в текст
    std::string tokens_to_text(const std::vector<int>& tokens) const {
        std::string text = "";
        for (int token : tokens) {
            text += decode(token);
        }
        return text;
    }
};
