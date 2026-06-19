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
#include "utils.h"

namespace {
using of::token_id;

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

std::vector<token_id> load_dataset(const std::string& path, vocabulary& vocab, std::size_t max_size = std::numeric_limits<std::size_t>::max()) {
	std::ifstream file(path, std::ios::binary);
	assert(file);
	std::vector<token_id> sample;
	for (char c; file.get(c) && sample.size() < max_size;) {
		sample.push_back(vocab.encode(static_cast<unsigned char>(c)));
	}
	return sample;
}


std::string generate(
	const of::oracle_forest& forest,
	const vocabulary& vocab,
	mx3random& rng,
	int num_tokens,
	double softmax_temperature) {
	std::vector<token_id> tokens{0};
	std::string text;
	for (int i = 0; i < num_tokens; ++i) {
		const auto probabilities = predict(forest, tokens, static_cast<uint32_t>(tokens.size() - 1), softmax_temperature);
		assert(!probabilities.empty());

		std::discrete_distribution<uint32_t> token_dist(probabilities.begin(), probabilities.end());
		const token_id next = token_dist(rng);
		text.push_back(vocab.decode(next));
		tokens.push_back(next);
	}
	return text;
}

std::vector<token_id> random_tokens(mx3random& r, uint32_t length, uint32_t vocab_size) {
	std::uniform_int_distribution<token_id> token_dist(0, vocab_size - 1);
	std::vector<token_id> tokens;
	tokens.reserve(length);
	for (uint32_t i = 0; i < length; ++i) {
		tokens.push_back(token_dist(r));
	}
	return tokens;
}

uint64_t no_change_hash(of::oracle_forest_config config) {
	constexpr uint32_t vocab_size = 20;
	mx3random r(1234);
	std::uniform_int_distribution<uint32_t> sample_size_dist(0, 50);

	std::vector<std::vector<token_id>> samples;
	samples.reserve(200);
	for (int i = 0; i < 200; ++i) {
		samples.push_back(random_tokens(r, sample_size_dist(r), vocab_size));
	}

	config.vocab_size = vocab_size;

	auto forest = of::build_oracle_forest(config, {samples});
	std::uniform_int_distribution<uint32_t> test_sample_size_dist(0, 60);
	uint64_t hash = 1;
	for (int i = 0; i < 1000; ++i) {
		const auto tokens = random_tokens(r, test_sample_size_dist(r), vocab_size);
		for (uint32_t pos = 0; pos < tokens.size(); ++pos) {
			for (const auto value : of::predict(forest, tokens, pos, 1.0)) {
				hash ^= std::hash<double>{}(value);
			}
		}
	}
	return hash;
}

void check_integrity() {
	constexpr uint64_t expected_hash = 13080308729460298394ull;

	const auto hash = no_change_hash({.context_size = 5, .max_depth = 8, .ensemble_size = 4});
	if (hash != expected_hash) {
		std::cout << "WARN: the new hash " << hash << " differs from expected hash " << expected_hash << ". Intentional behavior change?\n";
	}
}
} // namespace

int main(int argc, char** argv) {
	using namespace std::chrono_literals;

	constexpr std::size_t max_size = 500000000;

	const std::string path = argc > 1 ? argv[1] : "C:/tmp/datasets/tiny_stories/TinyStoriesV2.txt";
	//const std::string path = argc > 1 ? argv[1] : "C:/tmp/datasets/names/names.txt";
	//const std::string path = argc > 1 ? argv[1] : "C:/tmp/datasets/2800_books/1610.txt";
	//const std::string path = argc > 1 ? argv[1] : "C:/tmp/datasets/wiki_sentences/wikisent2.txt";

	check_integrity();

	vocabulary vocab;
	auto sample = load_dataset(path, vocab, max_size);
	if (sample.empty()) {
		std::cerr << "Failed to read any names from " << path << '\n';
		return 1;
	}
	std::cout << "Loaded '" << path << "' vocab size: " << vocab.size() << ", bytes: " << sample.size() << ".\n";

	of::oracle_forest_config cfg;
	cfg.vocab_size = vocab.size();
	cfg.max_depth = 4;
	cfg.ensemble_size = 4;

	evaluate_held_out_bpb(cfg, sample);


	std::cout << "Training...\n";
	auto train_start = std::chrono::steady_clock::now();
	auto forest = of::build_oracle_forest(cfg, {sample});
	auto train_end = std::chrono::steady_clock::now();
	const auto train_seconds = std::chrono::duration<double>(train_end - train_start).count();
	std::cout << "Training complete (" << train_seconds << " s).\n";
	const auto size_stats = size_bytes(forest);
	of::print_size_bytes(std::cout, size_stats);
	std::cout << "Model size ratio: " << static_cast<double>(size_stats.serialized) / sample.size() << "\n\n";


	std::cout << "Generating...\n";
	mx3random rng(42);
	auto generate_start = std::chrono::steady_clock::now();
	std::cout << generate(forest, vocab, rng, 1000, 1.) << '\n';
	auto generate_end = std::chrono::steady_clock::now();
	const auto generate_seconds = std::chrono::duration<double>(generate_end - generate_start).count();
	std::cout << "Generation time: " << generate_seconds << " s\n";
	return 0;
}
