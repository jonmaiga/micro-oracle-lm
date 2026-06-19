#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <span>
#include <unordered_map>
#include <vector>

#include "random.h"

namespace of {
using token_id = uint32_t;
using feature_id = uint64_t;
using context = std::vector<feature_id>;

struct oracle_leaf {
	struct prior_counts {
		uint32_t sample_count{};
		uint32_t feature_count{};
	};

	uint32_t sample_count{};
	std::vector<prior_counts> target_prior_counts;
	std::unordered_map<feature_id, std::unordered_map<token_id, uint32_t>> _feature_map; // feature -> target->count
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
	return static_cast<feature_id>(token) + static_cast<feature_id>(distance) * vocab_size;
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
		return (*tokens)[index + 1];
	}
};


inline context_view make_context_view(const std::vector<token_id>& tokens, uint32_t index,
                                      uint32_t context_size, uint32_t vocab_size) {
	const uint32_t count = index < context_size ? index : context_size;
	return {.tokens = &tokens, .view_size = count, .index = index, .vocab_size = vocab_size};
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
	assert(std::isfinite(temperature));
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

inline void train(oracle_leaf& leaf, const context_view& ctx, token_id target) {
	assert(target < leaf.target_prior_counts.size());

	++leaf.sample_count;
	++leaf.target_prior_counts[target].sample_count;
	leaf.target_prior_counts[target].feature_count += static_cast<uint32_t>(ctx.size());

	for (std::size_t i = 0; i < ctx.size(); ++i) {
		++leaf._feature_map[ctx[i]][target];
	}
}

inline std::vector<double> predict_logits(const oracle_leaf& leaf, const context_view& ctx, double smoothing) {
	assert(!leaf.target_prior_counts.empty());
	assert(leaf.sample_count > 0);

	const auto targets = leaf.target_prior_counts.size();
	std::vector<double> logits(targets);

	const double prior_den = static_cast<double>(leaf.sample_count) + smoothing * static_cast<double>(targets);
	for (std::size_t target = 0; target < targets; ++target) {
		const auto sample_count = leaf.target_prior_counts[target].sample_count;
		logits[target] = std::log((static_cast<double>(sample_count) + smoothing) / prior_den);
	}

	// todo: replace with global vocab_size?
	const auto vocabulary_size = static_cast<double>(leaf._feature_map.size());
	for (std::size_t i = 0; i < ctx.size(); ++i) {
		const auto it = leaf._feature_map.find(ctx[i]);
		if (it == leaf._feature_map.end()) {
			continue;
		}
		const auto& counts = it->second;
		for (token_id target = 0; target < targets; ++target) {
			const auto total = leaf.target_prior_counts[target].feature_count;
			const auto cit = counts.find(target);
			const auto count = cit != counts.end() ? cit->second : 0;
			const double den = static_cast<double>(total) + smoothing * vocabulary_size;
			logits[target] += std::log((static_cast<double>(count) + smoothing) / den);
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
	leaf.target_prior_counts.resize(cfg.vocab_size);
	for (const auto index : partition) {
		const auto& context = context_views[index];
		train(leaf, context, context.next_token());
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
	for (int i = 0; i < 7; ++i) {
		radius += context_distance(contexts[partition[dist(rng)]], center_ctx);
	}
	radius /= radius_samples;

	const auto split = std::ranges::stable_partition(partition, [&](const auto index) {
		return context_distance(contexts[index], center_ctx) < radius;
	});
	if (split.empty() || split.size() == count) {
		return build_leaf(tree, cfg, contexts, partition);
	}

	const auto mid = static_cast<std::size_t>(split.begin() - partition.begin());
	auto& nodes = tree.nodes;
	const auto node_index = nodes.size();
	nodes.push_back({.center_context = materialize(center_ctx), .radius = radius});
	nodes[node_index].inner = build_tree_recursively(tree, cfg, contexts, partition.first(mid), depth + 1, rng);
	nodes[node_index].outer = build_tree_recursively(tree, cfg, contexts, partition.subspan(mid), depth + 1, rng);
	return static_cast<uint32_t>(node_index);
}


inline oracle_tree build_tree(const oracle_forest_config& cfg, const std::vector<context_view>& contexts, std::vector<uint32_t> partition, mx3random& rng) {
	oracle_tree tree;
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


////
/// Build and predict
///

inline oracle_forest build_oracle_forest(const oracle_forest_config& cfg, const std::vector<std::vector<token_id>>& samples) {
	std::vector<std::vector<context_view>> token_contexts(cfg.vocab_size);
	for (const auto& sample : samples) {
		for (std::size_t i = 0; i + 1 < sample.size(); ++i) {
			token_contexts[sample[i]].push_back(make_context_view(sample, static_cast<uint32_t>(i), cfg.context_size, cfg.vocab_size));
		}
	}

	oracle_forest forest{.cfg = cfg};
	forest.trees.resize(cfg.vocab_size);
	const int token_count = static_cast<int>(forest.trees.size());
#pragma omp parallel for schedule(dynamic, 1)
	for (int token = 0; token < token_count; ++token) {
		const auto& contexts = token_contexts[token];
		if (contexts.empty()) {
			continue;
		}
		mx3random rng(token);
		std::vector<uint32_t> partition(contexts.size());
		std::iota(partition.begin(), partition.end(), 0u);
		auto& trees = forest.trees[token];
		trees.reserve(cfg.ensemble_size);

		for (uint32_t t = 0; t < cfg.ensemble_size; ++t) {
			trees.push_back(build_tree(cfg, contexts, partition, rng));
		}
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
