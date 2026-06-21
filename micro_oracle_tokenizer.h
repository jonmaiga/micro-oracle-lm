#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <omp.h>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "micro_oracle_lm.h"

namespace of {
// A subunit is a sequence of base token ids; these transparent hashers let the
// lookup table be probed with a std::span without materializing a vector per query.
struct token_seq_hash {
	using is_transparent = void;

	std::size_t operator()(std::span<const token_id> seq) const {
		std::size_t hash = 1469598103934665603ull; // FNV-1a offset basis
		for (const auto token : seq) {
			hash = (hash ^ token) * 1099511628211ull; // FNV-1a prime
		}
		return hash;
	}
};

struct token_seq_equal {
	using is_transparent = void;

	bool operator()(std::span<const token_id> lhs, std::span<const token_id> rhs) const {
		return std::ranges::equal(lhs, rhs);
	}
};

using subunit_counts = std::unordered_map<std::vector<token_id>, std::size_t, token_seq_hash, token_seq_equal>;

struct oracle_tokenizer_config {
	int max_subunits{1000};
	double surprise_bits{2.0};
};

struct oracle_tokenizer {
	std::vector<std::vector<token_id>> tokens; // token id -> subunit
	std::unordered_map<std::vector<token_id>, token_id, token_seq_hash, token_seq_equal> ids; // subunit -> token id
	std::vector<std::size_t> lookup_lengths; // distinct multi-token subunit lengths, descending
};

inline token_id add_token(oracle_tokenizer& tok, std::vector<token_id> subunit) {
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

inline bool by_count_desc(const std::pair<std::vector<token_id>, std::size_t>& lhs, const std::pair<std::vector<token_id>, std::size_t>& rhs) {
	if (lhs.second != rhs.second) {
		return lhs.second > rhs.second;
	}
	if (lhs.first.size() != rhs.first.size()) {
		return lhs.first.size() > rhs.first.size();
	}
	return lhs.first < rhs.first;
}

inline std::vector<std::vector<token_id>> select_top_subunits(const subunit_counts& counts, int max_subunits, std::size_t min_length = 1) {
	std::vector<std::pair<std::vector<token_id>, std::size_t>> ranked;
	ranked.reserve(counts.size());
	for (const auto& [subunit, count] : counts) {
		if (subunit.size() >= min_length) {
			ranked.emplace_back(subunit, count);
		}
	}
	// Always fully sort: by_count_desc is a total order on distinct sequences,
	// so this makes the result independent of unordered_map iteration order
	// (which is nondeterministic on MSVC due to randomized hashing).
	std::ranges::sort(ranked, by_count_desc);
	if (max_subunits >= 0 && ranked.size() > static_cast<std::size_t>(max_subunits)) {
		ranked.resize(static_cast<std::size_t>(max_subunits));
	}
	std::vector<std::vector<token_id>> subunits;
	subunits.reserve(ranked.size());
	for (auto& [subunit, _] : ranked) {
		subunits.push_back(std::move(subunit));
	}
	return subunits;
}

inline void count_subunit(subunit_counts& counts, const std::vector<token_id>& sample, std::size_t first, std::size_t last) {
	if (last - first >= 2) {
		++counts[std::vector<token_id>(sample.begin() + first, sample.begin() + last)];
	}
}

inline bool is_boundary(const oracle_forest& model, const std::vector<token_id>& sample, std::size_t index, double threshold) {
	const auto probabilities = predict(model, sample, static_cast<uint32_t>(index), 1.0);
	return probabilities[sample[index + 1]] < threshold;
}

inline subunit_counts measure_subunits(const oracle_tokenizer_config& cfg, const oracle_forest& model,
									   const std::vector<std::vector<token_id>>& samples) {
	const int thread_count = std::max(1, omp_get_max_threads());
	std::vector<subunit_counts> local_counts(thread_count);
	for (auto& local : local_counts) {
		local.reserve(samples.size() / thread_count + 1);
	}
	const auto threshold = std::pow(2.0, -cfg.surprise_bits);

#pragma omp parallel for schedule(dynamic, 1) num_threads(thread_count)
	for (int64_t s = 0; s < static_cast<int64_t>(samples.size()); ++s) {
		auto& local = local_counts[omp_get_thread_num()];
		const auto& sample = samples[s];
		std::size_t subunit_begin = 0;
		for (std::size_t i = 0; i + 1 < sample.size(); ++i) {
			if (is_boundary(model, sample, i, threshold)) {
				count_subunit(local, sample, subunit_begin, i + 1);
				subunit_begin = i + 1;
			}
		}
		count_subunit(local, sample, subunit_begin, sample.size());
	}

	subunit_counts counts;
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

inline oracle_tokenizer build_oracle_tokenizer(const oracle_tokenizer_config& cfg, const oracle_forest& model,
											   const std::vector<std::vector<token_id>>& samples) {
	oracle_tokenizer tok;
	if (samples.empty()) {
		return tok;
	}

	const auto counts = measure_subunits(cfg, model, samples);

	// Pre-add every base token as a single-token fallback so encoding never misses.
	for (token_id base = 0; base < model.cfg.vocab_size; ++base) {
		add_token(tok, std::vector<token_id>{base});
	}
	// Select only multi-token subunits (single tokens are already covered).
	for (auto& subunit : select_top_subunits(counts, cfg.max_subunits, 2)) {
		add_token(tok, std::move(subunit));
	}
	build_lookup_lengths(tok);
	return tok;
}

inline const std::pair<const std::vector<token_id>, token_id>* find_match(const oracle_tokenizer& tok, std::span<const token_id> tokens, std::size_t pos) {
	const auto remaining = tokens.size() - pos;
	for (const auto length : tok.lookup_lengths) {
		if (length > remaining) {
			continue;
		}
		const auto found = tok.ids.find(tokens.subspan(pos, length));
		if (found != tok.ids.end()) {
			return &*found;
		}
	}
	// Every base token is always present, so the single-token fallback never misses.
	const auto fallback = tok.ids.find(tokens.subspan(pos, 1));
	assert(fallback != tok.ids.end());
	return &*fallback;
}

inline std::vector<token_id> encode(const oracle_tokenizer& tok, std::span<const token_id> tokens) {
	std::vector<token_id> result;
	std::size_t pos = 0;
	while (pos < tokens.size()) {
		const auto* match = find_match(tok, tokens, pos);
		result.push_back(match->second);
		pos += match->first.size();
	}
	return result;
}
}
