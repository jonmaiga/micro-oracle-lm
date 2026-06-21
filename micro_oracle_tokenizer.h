#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <omp.h>
#include <span>
#include <unordered_map>
#include <vector>

#include "micro_oracle_lm.h"

namespace of {
using subunit = std::vector<token_id>;
using subunit_view = std::span<const token_id>;

// Transparent hash/equality lets subunit maps be probed with subunit_view without
// materializing a vector per query.
struct subunit_hash {
	using is_transparent = void;

	std::size_t operator()(subunit_view seq) const {
		std::size_t hash = 1469598103934665603ull; // FNV-1a offset basis
		for (const auto token : seq) {
			hash = (hash ^ token) * 1099511628211ull; // FNV-1a prime
		}
		return hash;
	}
};

struct subunit_equal {
	using is_transparent = void;

	bool operator()(subunit_view lhs, subunit_view rhs) const {
		return std::ranges::equal(lhs, rhs);
	}
};

using subunit_counts = std::unordered_map<subunit, std::size_t, subunit_hash, subunit_equal>;
using subunit_ids = std::unordered_map<subunit, token_id, subunit_hash, subunit_equal>;

struct oracle_tokenizer_config {
	std::size_t max_subunits{1000};
	double surprise_bits{2.0};
};

struct oracle_tokenizer {
	std::vector<subunit> tokens; // token id -> subunit
	subunit_ids ids; // subunit -> token id
	std::size_t max_length{1}; // longest subunit length (base tokens have length 1)
};

inline void add_token(oracle_tokenizer& tok, subunit unit) {
	assert(!tok.ids.contains(unit));

	const auto id = static_cast<token_id>(tok.tokens.size());
	tok.max_length = std::max(tok.max_length, unit.size());
	tok.ids.emplace(unit, id);
	tok.tokens.push_back(std::move(unit));
}

inline subunit_counts measure_subunits(const oracle_tokenizer_config& cfg, const oracle_forest& model,
                                       const std::vector<std::vector<token_id>>& samples) {
	const auto threshold = std::pow(2.0, -cfg.surprise_bits);

	// Map every predictable position (i, i+1 < sample.size()) of every sample onto
	// a single global index space. offsets[s] is the first global index of sample s.
	std::vector<std::size_t> offsets(samples.size() + 1);
	for (std::size_t s = 0; s < samples.size(); ++s) {
		const auto positions = samples[s].empty() ? 0 : samples[s].size() - 1;
		offsets[s + 1] = offsets[s] + positions;
	}
	const auto position_count = static_cast<int64_t>(offsets.back());

	// Phase 1: detect boundaries. Each position is independent and dominated by the
	// cost of predict(), so we parallelize over the flattened position space rather
	// than over samples. This keeps every thread busy even for a single huge sample.
	std::vector<uint8_t> boundaries(offsets.back());
#pragma omp parallel for schedule(dynamic, 256)
	for (int64_t g = 0; g < position_count; ++g) {
		const auto next = std::ranges::upper_bound(offsets, static_cast<std::size_t>(g));
		const auto s = static_cast<std::size_t>(next - offsets.begin()) - 1;
		const auto i = static_cast<std::size_t>(g) - offsets[s];
		const auto& sample = samples[s];
		const auto probabilities = predict(model, sample, static_cast<uint32_t>(i), 1.0);
		boundaries[g] = probabilities[sample[i + 1]] < threshold;
	}

	// Phase 2: slice samples at the detected boundaries and count subunits. This is
	// cheap relative to phase 1, so parallelizing over samples is sufficient.
	std::vector<subunit_counts> local_counts(omp_get_max_threads());

#pragma omp parallel for schedule(dynamic, 1)
	for (int64_t s = 0; s < static_cast<int64_t>(samples.size()); ++s) {
		auto& local = local_counts[omp_get_thread_num()];
		const auto& sample = samples[s];
		const auto base = offsets[s];
		const auto count_subunit = [&](std::size_t first, std::size_t last) {
			if (last - first >= 2) {
				++local[subunit(sample.begin() + first, sample.begin() + last)];
			}
		};

		std::size_t subunit_begin = 0;
		for (std::size_t i = 0; i + 1 < sample.size(); ++i) {
			if (boundaries[base + i]) {
				count_subunit(subunit_begin, i + 1);
				subunit_begin = i + 1;
			}
		}
		count_subunit(subunit_begin, sample.size());
	}

	subunit_counts counts;
	for (auto& local : local_counts) {
		for (auto& [subunit, count] : local) {
			counts[subunit] += count;
		}
	}
	return counts;
}

inline oracle_tokenizer build_oracle_tokenizer(const oracle_tokenizer_config& cfg, const oracle_forest& model,
                                               const std::vector<std::vector<token_id>>& samples) {
	const auto counts = measure_subunits(cfg, model, samples);
	std::vector<std::pair<subunit, std::size_t>> ranked(counts.begin(), counts.end());
	// Fully sort so the result is independent of unordered_map iteration order.
	std::ranges::sort(ranked, [](const auto& lhs, const auto& rhs) {
		if (lhs.second != rhs.second) {
			return lhs.second > rhs.second;
		}
		if (lhs.first.size() != rhs.first.size()) {
			return lhs.first.size() > rhs.first.size();
		}
		return lhs.first < rhs.first;
	});
	if (ranked.size() > cfg.max_subunits) {
		ranked.resize(cfg.max_subunits);
	}

	oracle_tokenizer tok;
	tok.tokens.reserve(model.cfg.vocab_size + ranked.size());
	tok.ids.reserve(model.cfg.vocab_size + ranked.size());
	for (token_id base = 0; base < model.cfg.vocab_size; ++base) {
		add_token(tok, subunit{base});
	}

	for (auto& [subunit, _] : ranked) {
		add_token(tok, std::move(subunit));
	}

	return tok;
}

inline std::vector<token_id> encode(const oracle_tokenizer& tok, subunit_view tokens) {
	std::vector<token_id> result;
	result.reserve(tokens.size());

	for (std::size_t pos = 0; pos < tokens.size();) {
		const auto max_length = std::min(tok.max_length, tokens.size() - pos);
		bool matched = false;
		for (std::size_t length = max_length; length >= 1; --length) {
			const auto found = tok.ids.find(tokens.subspan(pos, length));
			if (found != tok.ids.end()) {
				result.push_back(found->second);
				pos += length;
				matched = true;
				break;
			}
		}
		assert(matched && "length-1 fallback must always match");
	}
	return result;
}
}
