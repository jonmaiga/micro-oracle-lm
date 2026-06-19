#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thread>
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

std::vector<token_id> load_dataset(const std::string& path, vocabulary& vocab) {
	std::ifstream file(path, std::ios::binary);
	assert(file);
	std::vector<token_id> sample;
	for (char c; file.get(c);) {
		sample.push_back(vocab.encode(static_cast<unsigned char>(c)));
	}
	return sample;
}


// Samples a single name by repeatedly predicting the next token until the
std::string generate(
	const micro_oracle::config& config,
	const micro_oracle::oracle_forest& forest,
	const vocabulary& vocab,
	std::mt19937& rng, int num_tokens) {
	std::vector<token_id> tokens{0};
	std::string text;
	for (int i = 0; i < num_tokens; ++i) {
		const auto probabilities = predict(config, forest, tokens, static_cast<uint32_t>(tokens.size() - 1));
		assert(!probabilities.empty());

		std::discrete_distribution<uint32_t> dist(probabilities.begin(), probabilities.end());
		const token_id next = dist(rng);
		text.push_back(vocab.decode(next));
		tokens.push_back(next);
	}
	return text;
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

	auto forest = micro_oracle::train(config, {samples});

	uint64_t hash = 1;
	for (int i = 0; i < 1000; ++i) {
		const auto tokens = random_tokens(r, r.between(0, 60), 2 * vocab_size);
		for (uint32_t pos = 0; pos < tokens.size(); ++pos) {
			for (const auto value : micro_oracle::predict(config, forest, tokens, pos)) {
				hash ^= std::hash<double>{}(value);
			}
		}
	}
	return hash;
}

void check_integrity() {
	constexpr uint64_t expected_hash = 1205746795169421007ull;

	const auto hash = no_change_hash({.context_size = 5, .ensemble_size = 4, .max_depth = 8});
	if (hash != expected_hash) {
		std::cout << "WARN: the new hash " << hash << " differs from expected hash " << expected_hash << ". Intentional behavior change?\n";
	}
}

// Computes bits-per-byte (the average negative base-2 log-likelihood the model
// assigns to each actual next token) over the held-out tokens. Each token maps
// to one byte, so cross-entropy per token equals bits per byte.
double compute_bpb(const micro_oracle::config& cfg, const micro_oracle::oracle_forest& forest, const std::vector<token_id>& tokens) {
	constexpr double epsilon = 1e-12;
	const std::int64_t count = tokens.size() > 1 ? static_cast<std::int64_t>(tokens.size()) - 1 : 0;

	// predict() is a read-only, allocation-local operation, so positions can be
	// scored independently and summed via a reduction (matches the OpenMP
	// parallelism already used during training).
	double total_bits = 0.;
#pragma omp parallel for schedule(dynamic, 512) reduction(+ : total_bits)
	for (std::int64_t i = 0; i < count; ++i) {
		const auto probabilities = predict(cfg, forest, tokens, static_cast<uint32_t>(i));
		const token_id actual = tokens[i + 1];
		assert(actual < probabilities.size());
		const double p = probabilities[actual];
		total_bits += -std::log2(std::max(p, epsilon));
	}
	return count == 0 ? 0. : total_bits / static_cast<double>(count);
}

// Trains a fresh model on the first 80% of the sample and reports bits-per-byte
// on the remaining 20%, leaving the existing full-corpus training untouched.
void evaluate_held_out_bpb(const micro_oracle::config& base_cfg, const std::vector<token_id>& sample) {
	if (sample.size() < 2) {
		std::cerr << "Sample too small to evaluate held-out BPB.\n";
		return;
	}

	const std::size_t split = sample.size() * 4 / 5;
	const std::vector<token_id> train_tokens(sample.begin(), sample.begin() + split);
	const std::vector<token_id> test_tokens(sample.begin() + split, sample.end());
	std::cout << "Held-out evaluation: training on " << train_tokens.size()
		<< " bytes, testing on " << test_tokens.size() << " bytes.\n";

	auto train_start = std::chrono::steady_clock::now();
	const auto forest = train(base_cfg, {train_tokens});
	auto train_end = std::chrono::steady_clock::now();
	const auto train_seconds = std::chrono::duration<double>(train_end - train_start).count();

	auto eval_start = std::chrono::steady_clock::now();
	const double bpb = compute_bpb(base_cfg, forest, test_tokens);
	auto eval_end = std::chrono::steady_clock::now();
	const auto eval_seconds = std::chrono::duration<double>(eval_end - eval_start).count();

	std::cout << "Held-out BPB: " << bpb << " bits/byte (train " << train_seconds
		<< " s, eval " << eval_seconds << " s).\n";
}
} // namespace

int main(int argc, char** argv) {
	using namespace std::chrono_literals;
	//const std::string path = argc > 1 ? argv[1] : "C:/tmp/datasets/tiny_stories/TinyStoriesV2.txt";
	//const std::string path = argc > 1 ? argv[1] : "C:/tmp/datasets/names/names.txt";
	const std::string path = argc > 1 ? argv[1] : "C:/tmp/datasets/2800_books/1610.txt";
	//const std::string path = argc > 1 ? argv[1] : "C:/tmp/datasets/wiki_sentences/wikisent2.txt";

	check_integrity();

	vocabulary vocab;
	const auto sample = load_dataset(path, vocab);
	if (sample.empty()) {
		std::cerr << "Failed to read any names from " << path << '\n';
		return 1;
	}
	std::cout << "Loaded '" << path << "' vocab size: " << vocab.size() << ", bytes: " << sample.size() << ".\n";

	micro_oracle::config cfg;
	cfg.vocab_size = vocab.size();
	cfg.softmax_temperature = 1.;
	cfg.max_depth = 8;
	cfg.ensemble_size = 8;

	evaluate_held_out_bpb(cfg, sample);


	std::cout << "Training...\n";
	auto train_start = std::chrono::steady_clock::now();
	auto forest = micro_oracle::train(cfg, {sample});
	auto train_end = std::chrono::steady_clock::now();
	const auto train_seconds = std::chrono::duration<double>(train_end - train_start).count();
	std::cout << "Training complete (" << train_seconds << " s). Generating...\n";

	std::mt19937 rng(42);
	auto generate_start = std::chrono::steady_clock::now();
	std::cout << generate(cfg, forest, vocab, rng, 10000) << '\n';
	auto generate_end = std::chrono::steady_clock::now();
	const auto generate_seconds = std::chrono::duration<double>(generate_end - generate_start).count();
	std::cout << "Generation time: " << generate_seconds << " s\n";
	return 0;
}
