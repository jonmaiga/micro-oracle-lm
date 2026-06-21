#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include "random.h"

namespace of {
using token_id = uint32_t;
using feature_id = uint64_t;
using context = std::vector<feature_id>;

struct target_stat {
	uint32_t sample_count{};
	uint32_t feature_count{};
};

struct oracle_leaf {
	uint32_t sample_count{};
	std::vector<target_stat> target_stats;
	std::vector<feature_id> feature_keys; // index to feature offsets
	std::vector<uint32_t> feature_offsets; // [start, stop, ...] indices to entry + counts 
	std::vector<token_id> entry_targets;
	std::vector<uint32_t> entry_counts;
};

struct oracle_node {
	context center_context;
	double radius{};
	uint32_t inner{};
	uint32_t outer{};
	uint32_t leaf_index{std::numeric_limits<uint32_t>::max()};
};

inline bool is_leaf(const oracle_node& node) {
	return node.leaf_index != std::numeric_limits<uint32_t>::max();
}

struct oracle_tree {
	std::vector<oracle_node> nodes;
	std::vector<oracle_leaf> leaves;
};

struct oracle_forest_config {
	uint32_t vocab_size{};
	uint32_t context_size{6};
	uint32_t max_depth{8};
	uint32_t ensemble_size{8};
	double smoothing{0.5};
};

struct oracle_forest {
	oracle_forest_config cfg;
	std::vector<std::vector<oracle_tree>> trees;
};


inline feature_id make_feature(token_id token, uint32_t distance, uint32_t vocab_size) {
	const auto feature_vocab_size = static_cast<feature_id>(vocab_size) + 1;
	const auto feature_token = token < vocab_size ? static_cast<feature_id>(token) : static_cast<feature_id>(vocab_size);
	return feature_token + static_cast<feature_id>(distance) * feature_vocab_size;
}

struct context_view {
	const std::vector<token_id>* tokens{};
	uint32_t view_size{};
	uint32_t index{};
	uint32_t vocab_size{};

	std::size_t size() const {
		return view_size;
	}

	bool empty() const {
		return size() == 0;
	}

	feature_id operator[](std::size_t i) const {
		const uint32_t distance = static_cast<uint32_t>(i) + 1;
		return make_feature((*tokens)[index - distance], distance, vocab_size);
	}

	token_id next_token() const {
		assert(index + 1 < tokens->size());
		return (*tokens)[index + 1];
	}
};


inline context_view make_context_view(const std::vector<token_id>& tokens, uint32_t index,
                                      uint32_t context_size, uint32_t vocab_size) {
	return {.tokens = &tokens, .view_size = std::min(index, context_size), .index = index, .vocab_size = vocab_size};
}

inline context materialize(const context_view& view) {
	context ctx;
	ctx.reserve(view.size());
	for (std::size_t i = 0; i < view.size(); ++i) {
		ctx.push_back(view[i]);
	}
	return ctx;
}

inline std::vector<double> softmax_normalize(const std::vector<double>& logits, double temperature) {
	assert(temperature > 0);
	assert(!logits.empty());

	std::vector<double> probabilities(logits.size());
	const auto max_logit = *std::ranges::max_element(logits);
	double sum = 0.;
	for (std::size_t i = 0; i < logits.size(); ++i) {
		const auto v = std::exp((logits[i] - max_logit) / temperature);
		probabilities[i] = v;
		sum += v;
	}
	assert(sum > 0.0);
	assert(std::isfinite(sum));

	for (auto& v : probabilities) {
		v /= sum;
	}
	return probabilities;
}

template <class A, class B>
double context_distance(const A& a, const B& b) {
	if (a.empty() || b.empty()) {
		return a.empty() && b.empty() ? 0. : 1.;
	}

	std::size_t overlap = 0;
	for (std::size_t i = 0, j = 0; i < a.size() && j < b.size();) {
		assert(i == 0 || a[i - 1] <= a[i]);
		assert(j == 0 || b[j - 1] <= b[j]);

		if (a[i] == b[j]) {
			++overlap;
			++i;
			++j;
		}
		else if (a[i] < b[j]) {
			++i;
		}
		else {
			++j;
		}
	}
	const auto den = std::sqrt(static_cast<double>(a.size()) * static_cast<double>(b.size()));
	assert(den > 0);
	return 1. - static_cast<double>(overlap) / den;
}

inline void prepare_leaf_features(oracle_leaf& leaf, const std::vector<context_view>& context_views,
                                  std::span<const uint32_t> partition) {
	std::vector<std::pair<feature_id, token_id>> entries;
	std::size_t entry_count = 0;
	for (const auto index : partition) {
		entry_count += context_views[index].size();
	}
	entries.reserve(entry_count);

	for (const auto index : partition) {
		const auto& ctx = context_views[index];
		const auto target = ctx.next_token();
		for (std::size_t i = 0; i < ctx.size(); ++i) {
			entries.emplace_back(ctx[i], target);
		}
	}

	std::ranges::sort(entries);
	const auto last = std::ranges::unique(entries).begin();
	entries.erase(last, entries.end());

	leaf.feature_offsets.reserve(entries.size() + 1);
	leaf.feature_offsets.push_back(0);
	for (const auto& [feature, target] : entries) {
		if (leaf.feature_keys.empty() || leaf.feature_keys.back() != feature) {
			if (!leaf.feature_keys.empty()) {
				leaf.feature_offsets.push_back(static_cast<uint32_t>(leaf.entry_targets.size()));
			}
			leaf.feature_keys.push_back(feature);
		}
		leaf.entry_targets.push_back(target);
		leaf.entry_counts.push_back(0);
	}
	leaf.feature_offsets.push_back(static_cast<uint32_t>(leaf.entry_targets.size()));
}

inline void train(oracle_leaf& leaf, const context_view& ctx, token_id target) {
	assert(target < leaf.target_stats.size());

	++leaf.sample_count;
	++leaf.target_stats[target].sample_count;
	leaf.target_stats[target].feature_count += static_cast<uint32_t>(ctx.size());

	for (std::size_t i = 0; i < ctx.size(); ++i) {
		const auto key = ctx[i];
		const auto kit = std::ranges::lower_bound(leaf.feature_keys, key);
		assert(kit != leaf.feature_keys.end() && *kit == key);
		const auto feature_index = static_cast<std::size_t>(kit - leaf.feature_keys.begin());
		const auto entry_begin = leaf.entry_targets.begin() + leaf.feature_offsets[feature_index];
		const auto entry_end = leaf.entry_targets.begin() + leaf.feature_offsets[feature_index + 1];
		const auto tit = std::ranges::lower_bound(entry_begin, entry_end, target);
		assert(tit != entry_end && *tit == target);
		++leaf.entry_counts[static_cast<std::size_t>(tit - leaf.entry_targets.begin())];
	}
}

inline std::vector<double> predict_logits(const oracle_leaf& leaf, const context_view& ctx, double smoothing) {
	assert(!leaf.target_stats.empty());
	assert(leaf.sample_count > 0);

	const auto targets = leaf.target_stats.size();
	std::vector<double> logits(targets);

	const double prior_den = static_cast<double>(leaf.sample_count) + smoothing * static_cast<double>(targets);
	for (std::size_t target = 0; target < targets; ++target) {
		const auto sample_count = leaf.target_stats[target].sample_count;
		logits[target] = std::log((static_cast<double>(sample_count) + smoothing) / prior_den);
	}

	const double log_smoothing = std::log(smoothing);
	uint32_t present_features = 0;
	for (std::size_t i = 0; i < ctx.size(); ++i) {
		const auto key = ctx[i];
		const auto kit = std::ranges::lower_bound(leaf.feature_keys, key);
		if (kit == leaf.feature_keys.end() || *kit != key) {
			continue;
		}
		++present_features;
		const auto feature_index = static_cast<std::size_t>(kit - leaf.feature_keys.begin());
		const std::size_t entry_begin = leaf.feature_offsets[feature_index];
		const std::size_t entry_end = leaf.feature_offsets[feature_index + 1];
		// Nonzero CSR entries get this extra term over the zero-count baseline below.
		for (std::size_t j = entry_begin; j < entry_end; ++j) {
			logits[leaf.entry_targets[j]] += std::log(static_cast<double>(leaf.entry_counts[j]) + smoothing) - log_smoothing;
			assert(std::isfinite(logits[leaf.entry_targets[j]]));
		}
	}

	if (present_features != 0) {
		const auto unique_feature_count = static_cast<double>(leaf.feature_keys.size());
		const double scale = present_features;
		// Baseline: assume zero count for every matched context feature and target.
		for (std::size_t target = 0; target < targets; ++target) {
			const auto total = leaf.target_stats[target].feature_count;
			const double den = static_cast<double>(total) + smoothing * unique_feature_count;
			logits[target] += scale * (log_smoothing - std::log(den));
			assert(std::isfinite(logits[target]));
		}
	}

	return logits;
}


inline uint32_t build_leaf(oracle_tree& tree, const oracle_forest_config& cfg,
                           const std::vector<context_view>& context_views,
                           std::span<const uint32_t> partition) {
	const auto leaf_index = static_cast<uint32_t>(tree.leaves.size());
	auto& leaf = tree.leaves.emplace_back();
	leaf.target_stats.resize(cfg.vocab_size);
	prepare_leaf_features(leaf, context_views, partition);
	for (const auto index : partition) {
		const auto& ctx = context_views[index];
		train(leaf, ctx, ctx.next_token());
	}

	const auto node_index = tree.nodes.size();
	tree.nodes.push_back({.leaf_index = leaf_index});
	return static_cast<uint32_t>(node_index);
}

inline uint32_t build_tree_recursively(oracle_tree& tree, const oracle_forest_config& cfg,
                                       const std::vector<context_view>& contexts,
                                       std::span<uint32_t> partition, uint32_t depth, mx3random& rng) {
	const auto count = partition.size();
	if (depth >= cfg.max_depth || count < 2) {
		return build_leaf(tree, cfg, contexts, partition);
	}

	std::uniform_int_distribution<std::size_t> dist(0, count - 1);
	const auto center_ctx = contexts[partition[dist(rng)]];
	constexpr int radius_samples = 7;
	double radius = 0.;
	for (int i = 0; i < radius_samples; ++i) {
		radius += context_distance(contexts[partition[dist(rng)]], center_ctx);
	}
	radius /= radius_samples;

	const auto outer_partition = std::ranges::stable_partition(partition, [&](const auto index) {
		return context_distance(contexts[index], center_ctx) < radius;
	});
	if (outer_partition.empty() || outer_partition.size() == count) {
		return build_leaf(tree, cfg, contexts, partition);
	}

	const auto mid = static_cast<std::size_t>(outer_partition.begin() - partition.begin());
	const auto node_index = tree.nodes.size();
	tree.nodes.push_back({.center_context = materialize(center_ctx), .radius = radius});
	tree.nodes[node_index].inner = build_tree_recursively(tree, cfg, contexts, partition.first(mid), depth + 1, rng);
	tree.nodes[node_index].outer = build_tree_recursively(tree, cfg, contexts, partition.subspan(mid), depth + 1, rng);
	return static_cast<uint32_t>(node_index);
}


inline oracle_tree build_tree(const oracle_forest_config& cfg, const std::vector<context_view>& contexts, mx3random& rng) {
	oracle_tree tree;
	std::vector<uint32_t> partition(contexts.size());
	std::iota(partition.begin(), partition.end(), 0u);
	build_tree_recursively(tree, cfg, contexts, partition, 0, rng);
	return tree;
}

inline uint32_t route(const oracle_tree& tree, const context_view& ctx) {
	uint32_t node_index = 0;
	while (!is_leaf(tree.nodes[node_index])) {
		const auto& n = tree.nodes[node_index];
		node_index = context_distance(ctx, n.center_context) < n.radius ? n.inner : n.outer;
	}
	return node_index;
}

inline oracle_forest build_oracle_forest(const oracle_forest_config& cfg, const std::vector<std::vector<token_id>>& samples) {
	std::vector<std::vector<context_view>> token_contexts(cfg.vocab_size);
	for (const auto& sample : samples) {
		for (std::size_t i = 0; i + 1 < sample.size(); ++i) {
			if (sample[i] >= cfg.vocab_size || sample[i + 1] >= cfg.vocab_size) {
				continue;
			}
			token_contexts[sample[i]].push_back(make_context_view(sample, static_cast<uint32_t>(i), cfg.context_size, cfg.vocab_size));
		}
	}

	oracle_forest forest{.cfg = cfg};
	forest.trees.resize(cfg.vocab_size);
	const int token_count = static_cast<int>(forest.trees.size());
	for (int token = 0; token < token_count; ++token) {
		if (!token_contexts[token].empty()) {
			forest.trees[token].resize(cfg.ensemble_size);
		}
	}

	const auto task_count = static_cast<int64_t>(token_count) * static_cast<int64_t>(cfg.ensemble_size);
#pragma omp parallel for schedule(dynamic, 1)
	for (int64_t task = 0; task < task_count; ++task) {
		const auto token = static_cast<int>(task / cfg.ensemble_size);
		const auto& contexts = token_contexts[token];
		if (contexts.empty()) {
			continue;
		}

		const auto tree_index = static_cast<uint32_t>(task % cfg.ensemble_size);
		mx3random rng(task);
		forest.trees[token][tree_index] = build_tree(cfg, contexts, rng);
	}

	return forest;
}

inline std::vector<double> predict(const oracle_forest& forest, const std::vector<token_id>& tokens,
                                   uint32_t index_to_predict, double softmax_temperature) {
	assert(index_to_predict < tokens.size());

	const auto& cfg = forest.cfg;
	const auto current = tokens[index_to_predict];
	if (current >= forest.trees.size() || forest.trees[current].empty()) {
		// todo: oov
		return std::vector(cfg.vocab_size, 1. / cfg.vocab_size);
	}

	const auto ctx = make_context_view(tokens, index_to_predict, cfg.context_size, cfg.vocab_size);
	const auto& trees = forest.trees[current];

	std::vector<double> logits(cfg.vocab_size);
	for (const auto& tree : trees) {
		const auto node_index = route(tree, ctx);
		const auto& leaf = tree.leaves[tree.nodes[node_index].leaf_index];

		auto tree_logits = predict_logits(leaf, ctx, cfg.smoothing);
		std::ranges::transform(logits, tree_logits, logits.begin(), std::plus<double>{});
	}

	const auto inv = 1. / static_cast<double>(trees.size());
	for (auto& v : logits) {
		v *= inv;
	}

	return softmax_normalize(logits, softmax_temperature);
}
}
