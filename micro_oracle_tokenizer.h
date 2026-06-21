#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <omp.h>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "micro_oracle_lm.h"

namespace of {
struct transparent_string_hash {
	using is_transparent = void;

	std::size_t operator()(std::string_view sv) const {
		return std::hash<std::string_view>{}(sv);
	}
};

struct oracle_tokenizer_config {
	int max_subunits{4096};
	double surprise_bits{2.0};
	oracle_forest_config forest{};
};

struct oracle_tokenizer {
	std::vector<std::string> tokens; // token id -> subunit
	std::unordered_map<std::string, token_id, transparent_string_hash, std::equal_to<>> ids; // subunit -> token id
	std::vector<std::size_t> lookup_lengths; // distinct multi-byte subunit lengths, descending
};

inline token_id add_token(oracle_tokenizer& tok, std::string subunit) {
	const auto found = tok.ids.find(subunit);
	if (found != tok.ids.end()) {
		return found->second;
	}
	const auto id = static_cast<token_id>(tok.tokens.size());
	tok.ids.emplace(subunit, id);
	tok.tokens.push_back(std::move(subunit));
	return id;
}

inline void build_lookup_lengths(oracle_tokenizer& tok) {
	std::unordered_set<std::size_t> lengths;
	for (const auto& subunit : tok.tokens) {
		if (subunit.size() >= 2) {
			lengths.insert(subunit.size());
		}
	}
	tok.lookup_lengths.assign(lengths.begin(), lengths.end());
	std::ranges::sort(tok.lookup_lengths, std::greater<>());
}

inline bool by_count_desc(const std::pair<std::string, std::size_t>& lhs, const std::pair<std::string, std::size_t>& rhs) {
	if (lhs.second != rhs.second) {
		return lhs.second > rhs.second;
	}
	if (lhs.first.size() != rhs.first.size()) {
		return lhs.first.size() > rhs.first.size();
	}
	return lhs.first < rhs.first;
}

inline std::vector<std::string> select_top_subunits(const std::unordered_map<std::string, std::size_t>& counts, int max_subunits, std::size_t min_length = 1) {
	std::vector<std::pair<std::string, std::size_t>> ranked;
	ranked.reserve(counts.size());
	for (const auto& [subunit, count] : counts) {
		if (subunit.size() >= min_length) {
			ranked.emplace_back(subunit, count);
		}
	}
	// Always fully sort: by_count_desc is a total order on distinct strings,
	// so this makes the result independent of unordered_map iteration order
	// (which is nondeterministic on MSVC due to randomized string hashing).
	std::ranges::sort(ranked, by_count_desc);
	if (max_subunits >= 0 && ranked.size() > static_cast<std::size_t>(max_subunits)) {
		ranked.resize(static_cast<std::size_t>(max_subunits));
	}
	std::vector<std::string> subunits;
	subunits.reserve(ranked.size());
	for (auto& [subunit, _] : ranked) {
		subunits.push_back(std::move(subunit));
	}
	return subunits;
}

inline void count_subunit(std::unordered_map<std::string, std::size_t>& counts, const std::string& text, std::size_t first, std::size_t last) {
	if (last - first >= 2) {
		++counts[std::string(text.data() + first, last - first)];
	}
}

inline bool is_boundary(const oracle_forest& model, const std::vector<token_id>& sample, std::size_t index, double threshold) {
	const auto probabilities = predict(model, sample, static_cast<uint32_t>(index), 1.0);
	return probabilities[sample[index + 1]] < threshold;
}

inline std::unordered_map<std::string, std::size_t> measure_subunits(const oracle_tokenizer_config& cfg, const oracle_forest& model,
                                                                     const std::vector<std::string>& texts,
                                                                     const std::vector<std::vector<token_id>>& samples) {
	const int thread_count = std::max(1, omp_get_max_threads());
	std::vector<std::unordered_map<std::string, std::size_t>> local_counts(thread_count);
	for (auto& local : local_counts) {
		local.reserve(samples.size() / thread_count + 1);
	}
	const auto threshold = std::pow(2.0, -cfg.surprise_bits);

#pragma omp parallel for schedule(dynamic, 1) num_threads(thread_count)
	for (int64_t s = 0; s < static_cast<int64_t>(samples.size()); ++s) {
		auto& local = local_counts[omp_get_thread_num()];
		const auto& sample = samples[s];
		const auto& text = texts[s];
		std::size_t subunit_begin = 0;
		for (std::size_t i = 0; i + 1 < sample.size(); ++i) {
			if (is_boundary(model, sample, i, threshold)) {
				count_subunit(local, text, subunit_begin, i + 1);
				subunit_begin = i + 1;
			}
		}
		count_subunit(local, text, subunit_begin, sample.size());
	}

	std::unordered_map<std::string, std::size_t> counts;
	std::size_t total_unique = 0;
	for (const auto& local : local_counts) {
		total_unique += local.size();
	}
	counts.reserve(total_unique);
	for (auto& local : local_counts) {
		for (auto& [subunit, count] : local) {
			counts[subunit] += count;
		}
	}
	return counts;
}

inline oracle_tokenizer build_oracle_tokenizer(oracle_tokenizer_config cfg, const std::vector<std::string>& samples) {
	oracle_tokenizer tok;
	if (samples.empty()) {
		return tok;
	}

	// Char-level tokenization feeding the boundary model.
	vocabulary vocab;
	std::vector<std::vector<token_id>> char_samples;
	char_samples.reserve(samples.size());
	for (const auto& sample : samples) {
		auto& tokens = char_samples.emplace_back();
		tokens.reserve(sample.size());
		for (const unsigned char byte : sample) {
			tokens.push_back(vocab.encode(byte));
		}
	}

	cfg.forest.vocab_size = vocab.size();
	const auto model = build_oracle_forest(cfg.forest, char_samples);
	const auto counts = measure_subunits(cfg, model, samples, char_samples);

	// Pre-add all 256 single-byte fallback tokens.
	for (int b = 0; b < 256; ++b) {
		add_token(tok, std::string(1, static_cast<char>(b)));
	}
	// Select only multi-byte subunits (single-byte are already covered).
	for (const auto& subunit : select_top_subunits(counts, cfg.max_subunits, 2)) {
		add_token(tok, subunit);
	}
	build_lookup_lengths(tok);
	return tok;
}

inline const std::pair<const std::string, token_id>* find_match(const oracle_tokenizer& tok, std::string_view text, std::size_t pos) {
	const auto remaining = text.size() - pos;
	for (const auto length : tok.lookup_lengths) {
		if (length > remaining) {
			continue;
		}
		const auto found = tok.ids.find(text.substr(pos, length));
		if (found != tok.ids.end()) {
			return &*found;
		}
	}
	// All 256 single bytes are always present, so the fallback never misses.
	const auto fallback = tok.ids.find(text.substr(pos, 1));
	assert(fallback != tok.ids.end());
	return &*fallback;
}

inline std::vector<token_id> encode(const oracle_tokenizer& tok, std::string_view text) {
	std::vector<token_id> tokens;
	std::size_t pos = 0;
	while (pos < text.size()) {
		const auto* match = find_match(tok, text, pos);
		tokens.push_back(match->second);
		pos += match->first.size();
	}
	return tokens;
}
}
