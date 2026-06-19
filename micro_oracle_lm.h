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
	struct label_stat {
		uint32_t sample_total{};
		uint32_t feature_total{};
	};

	uint32_t sample_count{};
	std::vector<label_stat> label_stats;
	std::unordered_map<feature_id, std::unordered_map<token_id, uint32_t>> _feature_map; // feature -> next_token->count
};

struct oracle_node {
	bool leaf{true};
	context center_context;
	double radius{};
	uint32_t inner{};
	uint32_t outer{};
	uint32_t leaf_index{};
};

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

// Non-owning, lazily evaluated view over the context preceding a position in a
// token sequence (together with the label that follows it). Features are yielded
// in ascending-distance order and computed on access instead of stored, so the
// hot training path needs no per-event allocation. When an owning copy is
// required (e.g. a node center kept for routing) it can materialize() a context.
struct context_view {
	const std::vector<token_id>* tokens{};
	uint32_t index{};
	uint32_t context_size{};
	uint32_t vocab_size{};

	std::size_t size() const {
		return index < context_size ? index : context_size;
	}

	bool empty() const {
		return size() == 0;
	}

	feature_id operator[](std::size_t i) const {
		const uint32_t distance = static_cast<uint32_t>(i) + 1;
		return make_feature((*tokens)[index - distance], distance, vocab_size);
	}

	token_id next() const {
		return (*tokens)[index + 1];
	}

	context materialize() const {
		context ctx;
		const auto n = size();
		ctx.reserve(n);
		for (std::size_t i = 0; i < n; ++i) {
			ctx.push_back((*this)[i]);
		}
		return ctx;
	}
};

inline std::vector<double> softmax_normalize(const std::vector<double>& values, double temperature) {
	assert(temperature > 0);
	assert(std::isfinite(temperature));

	std::vector<double> result(values.size());
	const auto max_log = *std::ranges::max_element(values);
	double sum = 0.;
	for (std::size_t i = 0; i < values.size(); ++i) {
		const auto v = std::exp((values[i] - max_log) / temperature);
		result[i] = v;
		sum += v;
	}
	assert(sum > 0.0);
	assert(std::isfinite(sum));

	for (auto& v : result) {
		v /= sum;
	}
	return result;
}

// Cosine distance over the (binary) feature sets. Both feature lists are sorted
// by construction, so overlap is a linear merge. Templated so it works on both
// owning contexts and lazy context_views without forcing materialization.
// Named context_distance (not distance) to avoid ADL collision with std::distance.
template <class A, class B>
double context_distance(const A& a, const B& b) {
	if (a.empty() || b.empty()) {
		return a.empty() && b.empty() ? 0. : 1.;
	}

	std::size_t overlap = 0;
	for (std::size_t i = 0, j = 0; i < a.size() && j < b.size();) {
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

template <class Range>
void train(oracle_leaf& leaf, const Range& ctx, token_id label) {
	assert(label < leaf.label_stats.size());

	++leaf.sample_count;
	++leaf.label_stats[label].sample_total;
	leaf.label_stats[label].feature_total += static_cast<uint32_t>(ctx.size());

	for (std::size_t i = 0; i < ctx.size(); ++i) {
		++leaf._feature_map[ctx[i]][label];
	}
}

inline std::vector<double> predict(const oracle_leaf& leaf, const context& ctx, double smoothing) {
	std::vector<double> out(leaf.label_stats.size());

	assert(!leaf.label_stats.empty());
	assert(out.size() == leaf.label_stats.size());
	assert(leaf.sample_count > 0);

	const auto class_count = leaf.label_stats.size();

	const double prior_den = static_cast<double>(leaf.sample_count) + smoothing * static_cast<double>(class_count);
	for (std::size_t label = 0; label < class_count; ++label) {
		const auto count = leaf.label_stats[label].sample_total;
		out[label] = std::log((static_cast<double>(count) + smoothing) / prior_den);
	}

	// todo: replace with global vocab_size?
	const auto vocabulary_size = static_cast<double>(leaf._feature_map.size());
	for (const auto f : ctx) {
		const auto it = leaf._feature_map.find(f);
		if (it == leaf._feature_map.end()) {
			continue;
		}
		const auto& counts = it->second;
		for (token_id label = 0; label < class_count; ++label) {
			const auto total = leaf.label_stats[label].feature_total;
			const auto cit = counts.find(label);
			const auto count = cit != counts.end() ? cit->second : 0;
			const double den = static_cast<double>(total) + smoothing * vocabulary_size;
			out[label] += std::log((static_cast<double>(count) + smoothing) / den);
			assert(std::isfinite(out[label]));
		}
	}

	return out;
}


inline uint32_t build_leaf(oracle_tree& tree, const oracle_forest_config& cfg,
                           std::span<const context_view> events,
                           std::span<const uint32_t> source) {
	const auto leaf_index = static_cast<uint32_t>(tree.leaves.size());
	auto& leaf = tree.leaves.emplace_back();
	leaf.label_stats.resize(cfg.vocab_size);
	for (const auto index : source) {
		const auto& e = events[index];
		train(leaf, e, e.next());
	}

	const auto node_index = tree.nodes.size();
	tree.nodes.push_back({.leaf = true, .leaf_index = leaf_index});
	return static_cast<uint32_t>(node_index);
}

inline uint32_t build_tree_recursively(oracle_tree& tree, const oracle_forest_config& cfg,
                                       const std::vector<context_view>& contexts, std::span<uint32_t> event_indices, uint32_t depth, mx3random& rng) {
	const auto event_count = event_indices.size();
	if (depth >= cfg.max_depth || event_count < 2) {
		return build_leaf(tree, cfg, contexts, event_indices);
	}

	std::uniform_int_distribution<std::size_t> dist(0, event_count - 1);
	const auto center_ctx = contexts[event_indices[dist(rng)]].materialize();
	constexpr int radius_samples = 7;
	double radius = 0.;
	for (int i = 0; i < 7; ++i) {
		radius += context_distance(contexts[event_indices[dist(rng)]], center_ctx);
	}
	radius /= radius_samples;

	const auto split = std::ranges::stable_partition(event_indices, [&](const auto index) {
		return context_distance(contexts[index], center_ctx) < radius;
	});
	if (split.empty() || split.size() == event_count) {
		return build_leaf(tree, cfg, contexts, event_indices);
	}

	const auto mid = static_cast<std::size_t>(split.begin() - event_indices.begin());
	auto& nodes = tree.nodes;
	const auto node_index = nodes.size();
	nodes.push_back({.leaf = false, .center_context = center_ctx, .radius = radius});
	nodes[node_index].inner = build_tree_recursively(tree, cfg, contexts, event_indices.first(mid), depth + 1, rng);
	nodes[node_index].outer = build_tree_recursively(tree, cfg, contexts, event_indices.subspan(mid), depth + 1, rng);
	return static_cast<uint32_t>(node_index);
}


inline oracle_tree build_tree(const oracle_forest_config& cfg, const std::vector<context_view>& contexts, std::vector<uint32_t> indices, mx3random& rng) {
	oracle_tree tree;
	build_tree_recursively(tree, cfg, contexts, indices, 0, rng);
	return tree;
}

inline uint32_t route(const oracle_tree& tree, const context& ctx) {
	uint32_t node_index = 0;
	while (!tree.nodes[node_index].leaf) {
		const auto& n = tree.nodes[node_index];
		node_index = context_distance(ctx, n.center_context) < n.radius ? n.inner : n.outer;
	}
	return node_index;
}


////
/// Build and predict
///
///


inline oracle_forest build_oracle_forest(const oracle_forest_config& cfg, const std::vector<std::vector<token_id>>& samples) {
	oracle_forest forest{.cfg = cfg};
	forest.trees.resize(cfg.vocab_size);

	std::vector<std::vector<context_view>> events(cfg.vocab_size);
	for (const auto& sample : samples) {
		for (std::size_t i = 0; i + 1 < sample.size(); ++i) {
			events[sample[i]].push_back({&sample, static_cast<uint32_t>(i), cfg.context_size, cfg.vocab_size});
		}
	}

	const int token_count = static_cast<int>(forest.trees.size());
#pragma omp parallel for schedule(dynamic, 1)
	for (int token = 0; token < token_count; ++token) {
		const auto& contexts = events[token];
		if (contexts.empty()) {
			continue;
		}
		mx3random rng(token);
		std::vector<uint32_t> indices(contexts.size());
		std::iota(indices.begin(), indices.end(), 0u);
		auto& trees = forest.trees[token];
		trees.reserve(cfg.ensemble_size);

		for (uint32_t t = 0; t < cfg.ensemble_size; ++t) {
			trees.push_back(build_tree(cfg, contexts, indices, rng));
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

	const auto ctx = context_view{&tokens, index_to_predict, cfg.context_size, cfg.vocab_size}.materialize();
	const auto& trees = forest.trees[current];

	std::vector<double> probabilities(cfg.vocab_size);
	for (const auto& tree : trees) {
		const auto node_index = route(tree, ctx);
		const auto& leaf = tree.leaves[tree.nodes[node_index].leaf_index];

		auto tree_prediction = predict(leaf, ctx, cfg.smoothing);
		std::ranges::transform(probabilities, tree_prediction, probabilities.begin(), std::plus<double>{});
	}

	const auto inv = 1. / static_cast<double>(trees.size());
	for (auto& v : probabilities) {
		v *= inv;
	}

	return softmax_normalize(probabilities, softmax_temperature);
}
}
