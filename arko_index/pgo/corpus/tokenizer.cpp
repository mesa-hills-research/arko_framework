// Training sample for the arko_index PGO profiles: a tokenizer and a word counter.
// Written for this purpose; it is parsed, never compiled.
#include "lru_cache.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace arko::sample {
namespace {

constexpr char kSeparators[] = " \t\r\n,;:.!?()[]{}\"'";
constexpr int kMaxTokenLength = 0x100;

bool is_separator(char c) {
    for (const char* p = kSeparators; *p != '\0'; ++p) {
        if (*p == c) return true;
    }
    return false;
}

}  // namespace

enum class TokenKind : unsigned char { Word, Number, Symbol };

struct Token {
    TokenKind kind;
    std::string text;
    std::size_t offset;
    friend bool operator==(const Token&, const Token&) = default;
};

class Tokenizer {
public:
    explicit Tokenizer(std::string_view input) : input_(input) {}
    virtual ~Tokenizer() = default;

    std::vector<Token> all() {
        std::vector<Token> out;
        out.reserve(input_.size() / 4);
        while (auto token = next()) {
            out.push_back(*std::move(token));
        }
        return out;
    }

    virtual std::optional<Token> next() {
        while (pos_ < input_.size() && is_separator(input_[pos_])) ++pos_;
        if (pos_ >= input_.size()) return std::nullopt;
        const std::size_t start = pos_;
        TokenKind kind = std::isdigit(static_cast<unsigned char>(input_[pos_])) ? TokenKind::Number
                         : std::isalpha(static_cast<unsigned char>(input_[pos_])) ? TokenKind::Word
                                                                                  : TokenKind::Symbol;
        while (pos_ < input_.size() && !is_separator(input_[pos_])) {
            if (pos_ - start >= kMaxTokenLength) {
                throw std::length_error("token too long at offset " + std::to_string(start));
            }
            ++pos_;
        }
        return Token{kind, std::string(input_.substr(start, pos_ - start)), start};
    }

protected:
    std::string_view input_;
    std::size_t pos_ = 0;
};

class LowercaseTokenizer final : public Tokenizer {
public:
    using Tokenizer::Tokenizer;
    std::optional<Token> next() override {
        auto token = Tokenizer::next();
        if (token && token->kind == TokenKind::Word) {
            std::transform(token->text.begin(), token->text.end(), token->text.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        }
        return token;
    }
};

std::map<std::string, int> count_words(std::istream& in, LruCache<std::string, int>* cache = nullptr) {
    std::map<std::string, int> counts;
    std::string line;
    while (std::getline(in, line)) {
        auto tokenizer = std::make_unique<LowercaseTokenizer>(line);
        for (auto& token : tokenizer->all()) {
            switch (token.kind) {
                case TokenKind::Word:
                    ++counts[token.text];
                    break;
                case TokenKind::Number:
                case TokenKind::Symbol:
                default:
                    continue;
            }
            if (cache != nullptr) cache->put(token.text, counts[token.text]);
        }
    }
    return counts;
}

}  // namespace arko::sample

int main(int argc, char** argv) {
    using namespace arko::sample;
    std::istringstream text{R"(The quick brown fox jumps over the lazy dog. 42 times; the end!)"};
    LruCache<std::string, int> cache(8);
    const auto counts = count_words(text, &cache);
    for (const auto& [word, n] : counts) {
        std::cout << word << '\t' << n << "\n";
    }
    std::cerr << describe(cache.stats()) << std::endl;
    return argc > 1 && argv[1][0] == '-' ? 1 : 0;
}
