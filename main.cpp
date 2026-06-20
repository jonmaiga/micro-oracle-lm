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


std::vector<unsigned char> load_dataset(const std::string& path, std::size_t max_size = std::numeric_limits<std::size_t>::max()) {
	std::ifstream file(path, std::ios::binary);
	assert(file);
	std::vector<unsigned char> sample;
	for (char c; file.get(c) && sample.size() < max_size;) {
		sample.push_back(c);
	}
	return sample;
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
		const auto tokens = random_tokens(r, test_sample_size_dist(r), vocab_size + 10);
		for (uint32_t pos = 0; pos < tokens.size(); ++pos) {
			for (const auto value : of::predict(forest, tokens, pos, 1.0)) {
				hash ^= std::hash<double>{}(value);
			}
		}
	}
	return hash;
}

void check_integrity() {
	constexpr uint64_t expected_hash = 155657146846925995ull;

	const auto hash = no_change_hash({.context_size = 5, .max_depth = 8, .ensemble_size = 4});
	if (hash != expected_hash) {
		std::cout << "WARN: the new hash " << hash << " differs from expected hash " << expected_hash << ". Intentional behavior change?\n";
	}
}
} // namespace

int main(int argc, char** argv) {
	using namespace std::chrono_literals;

	check_integrity();

	constexpr std::size_t max_size = 500000000;

	const std::string base_path = "C:/tmp/datasets/";

	//const std::string path = argc > 1 ? argv[1] : "C:/tmp/datasets/tiny_stories/TinyStoriesV2.txt";
	//const std::string path = argc > 1 ? argv[1] : "C:/tmp/datasets/names/names.txt";
	//const std::string path = argc > 1 ? argv[1] : "C:/tmp/datasets/tiny_shakespeare/train.txt";
	//const std::string path = argc > 1 ? argv[1] : "C:/tmp/datasets/2800_books/1610.txt";
	//const std::string path = argc > 1 ? argv[1] : "C:/tmp/datasets/wiki_sentences/wikisent2.txt";
	const std::string path = argc > 1 ? argv[1] : "C:/tmp/datasets/fineweb_ultra_mini/2013_20_001.csv";

	auto sample = load_dataset(path, max_size);
	//auto s2 = load_dataset("C:/tmp/datasets/tiny_stories/TinyStoriesV2.txt", sample.size());
	//auto s3 = load_dataset("C:/tmp/datasets/tiny_shakespeare/train.txt", sample.size());
	//auto s4 = load_dataset("C:/tmp/datasets/fineweb_ultra_mini/2013_20_001.csv", sample.size());
	//sample.insert(sample.end(), s2.begin(), s2.end());
	//sample.insert(sample.end(), s3.begin(), s3.end());
	//sample.insert(sample.end(), s4.begin(), s4.end());


	if (sample.empty()) {
		std::cerr << "Failed to read sample from " << path << '\n';
		return 1;
	}
	std::cout << "Loaded '" << path << " bytes: " << sample.size() << ".\n";

	of::oracle_forest_config cfg;
	cfg.max_depth = 8;
	cfg.ensemble_size = 8;
	cfg.context_size = 6;
	evaluate_held_out_bpb(cfg, sample);

	return 0;
}
