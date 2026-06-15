#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "micro_oracle_lm.h"

namespace {
using micro_oracle::token_id;

class vocabulary {
public:
	token_id encode(unsigned char c) {
		auto& slot = _char_to_token[c];
		if (slot == 0) {
			slot = static_cast<token_id>(_token_to_char.size());
			_token_to_char.push_back(static_cast<char>(c));
		}
		return slot;
	}

	char decode(token_id token) const {
		return _token_to_char[token];
	}

	uint32_t size() const {
		return static_cast<uint32_t>(_token_to_char.size());
	}

private:
	std::array<token_id, 256> _char_to_token{};
	std::vector<char> _token_to_char;
};

std::vector<std::vector<token_id>> load_dataset(const std::string& path, vocabulary& vocab) {
	std::ifstream file(path, std::ios::binary);
	std::vector<token_id> sample;
	for (char c; file.get(c);) {
		sample.push_back(vocab.encode(static_cast<unsigned char>(c)));
	}
	return {std::move(sample)};
}


// Samples a single name by repeatedly predicting the next token until the
std::string generate(const micro_oracle::micro_oracle_lm& model, const vocabulary& vocab,
                     std::mt19937& rng, int num_tokens) {
	std::vector<token_id> tokens{0};
	std::string name;
	for (int i = 0; i < num_tokens; ++i) {
		const auto probabilities = model.predict(tokens, static_cast<uint32_t>(tokens.size() - 1));
		assert(!probabilities.empty());

		std::discrete_distribution<uint32_t> dist(probabilities.begin(), probabilities.end());
		const token_id next = dist(rng);
		name.push_back(vocab.decode(next));
		tokens.push_back(next);
	}
	return name;
}

class random {
public:
	explicit random(uint64_t seed) : _state(seed) {
	}

	uint64_t next() {
		_state ^= _state << 13;
		_state ^= _state >> 7;
		_state ^= _state << 17;
		return _state;
	}

	uint32_t between(uint32_t lo, uint32_t hi) {
		return lo + static_cast<uint32_t>(next() % (hi - lo + 1));
	}

private:
	uint64_t _state{};
};

std::vector<token_id> random_tokens(random& r, uint32_t length, uint32_t vocab_size) {
	std::vector<token_id> tokens;
	tokens.reserve(length);
	for (uint32_t i = 0; i < length; ++i) {
		tokens.push_back(r.between(0, vocab_size - 1));
	}
	return tokens;
}

uint64_t no_change_hash(micro_oracle::config config) {
	constexpr uint32_t vocab_size = 20;
	random r(1234);

	std::vector<std::vector<token_id>> samples;
	samples.reserve(200);
	for (int i = 0; i < 200; ++i) {
		samples.push_back(random_tokens(r, r.between(0, 50), vocab_size));
	}

	config.vocab_size = vocab_size;

	micro_oracle::micro_oracle_lm model(config);
	model.train(samples);

	uint64_t hash = 1;
	for (int i = 0; i < 1000; ++i) {
		const auto tokens = random_tokens(r, r.between(0, 60), 2 * vocab_size);
		for (uint32_t pos = 0; pos < tokens.size(); ++pos) {
			for (const auto value : model.predict(tokens, pos)) {
				hash ^= std::hash<double>{}(value);
			}
		}
	}
	return hash;
}

void check_integrity() {
	constexpr uint64_t expected_hash = 4540134499834516908ull;

	const auto hash = no_change_hash({.context_size = 5, .ensemble_size = 4, .max_depth = 8});
	if (hash != expected_hash) {
		std::cout << "WARN: the new hash " << hash << " differs from expected hash " << expected_hash << ". Intentional behavior change?\n";
	}
}
} // namespace

int main(int argc, char** argv) {
	const std::string path = argc > 1 ? argv[1] : "C:/tmp/datasets/names/names.txt";

	check_integrity();

	vocabulary vocab;
	const auto samples = load_dataset(path, vocab);
	if (samples.empty()) {
		std::cerr << "Failed to read any names from " << path << '\n';
		return 1;
	}
	std::cout << "Loaded " << samples.size() << " names (vocab size " << vocab.size() << ").\n";

	micro_oracle::config cfg;
	cfg.vocab_size = vocab.size();
	micro_oracle::micro_oracle_lm model(cfg);

	std::cout << "Training...\n";
	model.train(samples);
	std::cout << "Training complete. Generating:\n\n";

	std::mt19937 rng(42);
	std::cout << generate(model, vocab, rng, 1000) << '\n';
	return 0;
}
