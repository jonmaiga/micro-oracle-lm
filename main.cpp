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

// Token id 0 is reserved as the sequence boundary (start/stop) marker.
constexpr token_id boundary = 0;

// Maps characters to contiguous token ids (boundary keeps id 0) and remembers
// the reverse mapping so generated tokens can be turned back into text.
class vocabulary {
public:
	vocabulary() {
		_token_to_char.push_back('\0'); // placeholder for the boundary token.
	}

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

// Reads one name per line and encodes it as [boundary, boundary, chars..., boundary].
// The two leading boundaries let the model learn the first-character distribution.
std::vector<std::vector<token_id>> load_names(const std::string& path, vocabulary& vocab) {
	std::ifstream in(path);
	if (!in) {
		return {};
	}

	std::vector<std::vector<token_id>> samples;
	std::string line;
	while (std::getline(in, line)) {
		while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ')) {
			line.pop_back();
		}
		if (line.empty()) {
			continue;
		}
		std::vector<token_id> sample;
		sample.reserve(line.size() + 3);
		sample.push_back(boundary);
		sample.push_back(boundary);
		for (const unsigned char c : line) {
			sample.push_back(vocab.encode(c));
		}
		sample.push_back(boundary);
		samples.push_back(std::move(sample));
	}
	return samples;
}

// Samples a single name by repeatedly predicting the next token until the
// boundary (stop) token is drawn or the length cap is reached.
std::string generate_name(const micro_oracle::micro_oracle_lm& model, const vocabulary& vocab,
                          std::mt19937& rng, int max_length) {
	std::vector<token_id> tokens{boundary, boundary};
	std::string name;
	for (int step = 0; step < max_length; ++step) {
		const auto probabilities = model.predict(tokens, static_cast<uint32_t>(tokens.size() - 1));
		if (probabilities.empty()) {
			break;
		}
		std::discrete_distribution<uint32_t> dist(probabilities.begin(), probabilities.end());
		const token_id next = dist(rng);
		if (next == boundary) {
			break;
		}
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
	constexpr uint32_t vocab_size = 16;
	random r(1234);

	std::vector<std::vector<token_id>> samples;
	samples.reserve(200);
	for (int i = 0; i < 200; ++i) {
		samples.push_back(random_tokens(r, 50, vocab_size));
	}

	config.vocab_size = vocab_size;
	micro_oracle::micro_oracle_lm model(config);
	model.train(samples);

	uint64_t hash = 1;
	for (int i = 0; i < 1000; ++i) {
		const auto tokens = random_tokens(r, r.between(1, 20), vocab_size);
		for (uint32_t j = 0; j < tokens.size(); ++j) {
			const auto p = model.predict(tokens, j);
			for (const auto value : p) {
				hash ^= std::hash<double>{}(value);
			}
		}
	}
	return hash;
}

void check_integrity() {
	constexpr uint64_t expected_hash = 1852781300469497625ull;
	const auto hash = no_change_hash({.context_size = 5, .ensemble_size = 4, .max_depth = 8});

	std::printf("hash = %lluull\n", static_cast<unsigned long long>(hash));
	if (expected_hash != 0 && hash != expected_hash) {
		std::printf("FAIL: expected %lluull\n", static_cast<unsigned long long>(expected_hash));
	}
	std::printf("OK\n");
}
} // namespace

int main(int argc, char** argv) {
	const std::string path = argc > 1 ? argv[1] : "C:/tmp/datasets/names/names.txt";

	check_integrity();

	vocabulary vocab;
	const auto samples = load_names(path, vocab);
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
	std::cout << "Training complete. Generating 100 names:\n\n";

	std::mt19937 rng(42);
	constexpr int num_names = 100;
	constexpr int max_length = 32;
	for (int i = 0; i < num_names; ++i) {
		std::cout << generate_name(model, vocab, rng, max_length) << '\n';
	}

	return 0;
}
