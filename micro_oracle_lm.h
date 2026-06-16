// micro_oracle_lm.h
//
// Minimal, self-contained single-header version of the fast HSROE language
// model. One small ensemble of ball-trees is built per current token. Each
// tree recursively splits its training events by cosine distance to a random
// center context; leaves hold Naive-Bayes label/feature counts. Prediction
// routes the context through every tree, averages the per-tree log-probs, and
// applies a softmax.
//
// Standalone: depends only on the C++ standard library. No introspection,
// ensemble routing hashes, or debug output.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <random>
#include <span>
#include <unordered_map>
#include <vector>

namespace micro_oracle {
using token_id = uint32_t;
using feature_id = uint64_t;

struct config {
	uint32_t vocab_size{};
	uint32_t context_size{6};
	uint32_t ensemble_size{8};
	uint32_t max_depth{8};
	double smoothing{0.5};
	double softmax_temperature{1.};
};

// A context is the sorted list of feature ids preceding the current token.
struct context {
	std::vector<feature_id> features;
};

// previous context + current token -> next token
struct event {
	context ctx;
	token_id next{};
};

// Encodes "previous token at a given backward distance" into one id.
inline feature_id make_feature(token_id token, uint32_t distance, uint32_t vocab_size) {
	return static_cast<feature_id>(token) + static_cast<feature_id>(distance) * vocab_size;
}

inline context make_context_at(const std::vector<token_id>& tokens, uint32_t index,
                               uint32_t context_size, uint32_t vocab_size) {
	context ctx;
	const auto start = index > context_size ? index - context_size : 0;
	ctx.features.reserve(index - start);
	for (uint32_t distance = 1; distance <= index - start; ++distance) {
		ctx.features.push_back(make_feature(tokens[index - distance], distance, vocab_size));
	}
	return ctx;
}

// Cosine distance over the (binary) feature sets. Both feature lists are sorted
// by construction, so overlap is a linear merge.
inline double distance(const context& a, const context& b) {
	if (a.features.empty() || b.features.empty()) {
		return a.features.empty() && b.features.empty() ? 0. : 1.;
	}

	std::size_t i = 0;
	std::size_t j = 0;
	std::size_t overlap = 0;
	while (i < a.features.size() && j < b.features.size()) {
		if (a.features[i] == b.features[j]) {
			++overlap;
			++i;
			++j;
		}
		else if (a.features[i] < b.features[j]) {
			++i;
		}
		else {
			++j;
		}
	}
	const auto den = std::sqrt(static_cast<double>(a.features.size()) * static_cast<double>(b.features.size()));
	assert(den > 0);
	return 1. - static_cast<double>(overlap) / den;
}

class leaf_model {
public:
	explicit leaf_model(uint32_t vocab_size) : _vocab_size(vocab_size) {
	}

	void observe(const context& ctx, token_id label) {
		assert(label < _vocab_size);

		++_sample_count;
		++_label_count[label].count;
		for (const auto f : ctx.features) {
			auto& counts = _feature_map[f];
			if (counts.empty()) {
				++_observed_feature_count;
			}
			++counts[label];
			++_label_count[label].feature_total;
		}
	}

	void predict_into(const context& ctx, std::vector<double>& out, double smoothing) const {
		assert(!_label_count.empty());
		assert(out.size() == _label_count.size());
		assert(_sample_count > 0);

		std::ranges::fill(out, 0.);
		const auto class_count = _label_count.size();

		const double prior_den = static_cast<double>(_sample_count) + smoothing * static_cast<double>(class_count);
		for (std::size_t label = 0; label < class_count; ++label) {
			const auto count = _label_count[label].count;
			out[label] = std::log((static_cast<double>(count) + smoothing) / prior_den);
		}

		// todo: replace with global vocab_size?
		// todo: max is needed because we might train with empty samples now, e.g. when there is not prev context
		const auto vocabulary_size = std::max<uint32_t>(1, _observed_feature_count);
		for (const auto f : ctx.features) {
			const auto it = _feature_map.find(f);
			if (it == _feature_map.end()) {
				continue;
			}
			const auto& counts = it->second;
			for (std::size_t label = 0; label < class_count; ++label) {
				const auto total = _label_count[label].feature_total;
				const auto cit = counts.find(static_cast<token_id>(label));
				const auto count = cit != counts.end() ? cit->second : 0;
				const double den = static_cast<double>(total) + smoothing * static_cast<double>(vocabulary_size);
				out[label] += std::log((static_cast<double>(count) + smoothing) / den);
				assert(std::isfinite(out[label]));
			}
		}
	}

private:
	struct label_stat {
		uint32_t count{};
		uint32_t feature_total{};
	};

	uint32_t _vocab_size{};
	uint32_t _sample_count{};
	uint32_t _observed_feature_count{};
	std::vector<label_stat> _label_count{std::vector<label_stat>(_vocab_size)};
	std::unordered_map<feature_id, std::unordered_map<token_id, uint32_t>> _feature_map;
};

class tree {
public:
	tree() = default;

	tree(const config& cfg, uint64_t seed, const std::vector<event>* events) :
		_cfg(cfg), _seed(seed), _events(events) {
	}

	void build(std::vector<uint32_t> event_indices) {
		_nodes.clear();
		_leaves.clear();
		build_node(event_indices, 0);
	}

	void predict_into(const context& ctx, std::vector<double>& out) const {
		const auto node_index = route(ctx);
		_leaves[_nodes[node_index].leaf_index].predict_into(ctx, out, _cfg.smoothing);
	}

private:
	struct node {
		bool leaf{true};
		uint32_t depth{};
		context center_context;
		double radius{};
		uint32_t inner{};
		uint32_t outer{};
		uint32_t leaf_index{};
	};

	uint32_t route(const context& ctx) const {
		uint32_t node_index = 0;
		while (!_nodes[node_index].leaf) {
			const auto& n = _nodes[node_index];
			node_index = distance(ctx, n.center_context) < n.radius ? n.inner : n.outer;
		}
		return node_index;
	}

	uint32_t build_node(std::span<uint32_t> event_indices, uint32_t depth) {
		const auto event_count = event_indices.size();
		if (depth >= _cfg.max_depth || event_count < 2) {
			return build_leaf(event_indices, depth);
		}

		std::uniform_int_distribution<std::size_t> dist(0, event_count - 1);
		std::mt19937_64 rng(_seed + depth * 16777619ull + event_count * 2166136261ull + _nodes.size());
		const auto& center_ctx = (*_events)[event_indices[dist(rng)]].ctx;
		const auto radius = sample_radius(event_indices, center_ctx, dist, rng);
		const auto split = std::ranges::stable_partition(event_indices, [&](const auto index) {
			return distance((*_events)[index].ctx, center_ctx) < radius;
		});
		if (split.empty() || split.size() == event_count) {
			return build_leaf(event_indices, depth);
		}

		const auto mid = static_cast<std::size_t>(split.begin() - event_indices.begin());
		const auto node_index = _nodes.size();
		_nodes.push_back({.leaf = false, .depth = depth, .center_context = center_ctx, .radius = radius});
		_nodes[node_index].inner = build_node(event_indices.first(mid), depth + 1);
		_nodes[node_index].outer = build_node(event_indices.subspan(mid), depth + 1);
		return static_cast<uint32_t>(node_index);
	}

	uint32_t build_leaf(std::span<const uint32_t> source, uint32_t depth) {
		const auto leaf_index = static_cast<uint32_t>(_leaves.size());
		auto& leaf = _leaves.emplace_back(_cfg.vocab_size);
		for (const auto index : source) {
			const auto& e = (*_events)[index];
			leaf.observe(e.ctx, e.next);
		}

		_nodes.push_back({.leaf = true, .depth = depth, .leaf_index = leaf_index});
		return static_cast<uint32_t>(_nodes.size() - 1);
	}

	double sample_radius(std::span<const uint32_t> source, const context& center_ctx,
	                     std::uniform_int_distribution<std::size_t>& dist, std::mt19937_64& rng) const {
		double radius = 0.;
		constexpr int radius_samples = 7;
		for (int i = 0; i < radius_samples; ++i) {
			radius += distance((*_events)[source[dist(rng)]].ctx, center_ctx);
		}
		return radius / radius_samples;
	}

	config _cfg;
	uint64_t _seed{};
	std::vector<node> _nodes;
	std::vector<leaf_model> _leaves;
	const std::vector<event>* _events{};
};

class micro_oracle_lm {
public:
	explicit micro_oracle_lm(const config& cfg) : _cfg(cfg) {
	}

	void train(const std::vector<std::vector<token_id>>& samples) {
		_models.assign(_cfg.vocab_size, {});

		for (const auto& sample : samples) {
			for (std::size_t i = 0; i + 1 < sample.size(); ++i) {
				_models[sample[i]].events.push_back({
					make_context_at(sample, static_cast<uint32_t>(i), _cfg.context_size, _cfg.vocab_size),
					sample[i + 1]
				});
			}
		}

		for (token_id token = 0; token < _models.size(); ++token) {
			auto& model = _models[token];
			if (model.events.empty()) {
				continue;
			}
			std::vector<uint32_t> indices(model.events.size());
			for (uint32_t i = 0; i < indices.size(); ++i) {
				indices[i] = i;
			}
			model.trees.resize(_cfg.ensemble_size);
			for (uint32_t t = 0; t < _cfg.ensemble_size; ++t) {
				model.trees[t] = tree(_cfg, static_cast<uint64_t>(token) * 1009ull + t, &model.events);
				model.trees[t].build(indices);
			}
		}
	}

	std::vector<double> predict(const std::vector<token_id>& tokens, uint32_t index_to_predict) const {
		assert(index_to_predict < tokens.size());

		if (tokens.empty()) {
			// todo: model prior
			return std::vector(_cfg.vocab_size, 1. / _cfg.vocab_size);
		}

		const auto current = tokens[index_to_predict];
		if (current >= _models.size() || _models[current].trees.empty()) {
			// todo: oov
			return std::vector(_cfg.vocab_size, 1. / _cfg.vocab_size);
		}

		const auto ctx = make_context_at(tokens, index_to_predict, _cfg.context_size, _cfg.vocab_size);
		const auto& model = _models[current];

		std::vector<double> probabilities(_cfg.vocab_size, 0.);
		std::vector<double> local(_cfg.vocab_size, 0.);
		for (const auto& t : model.trees) {
			t.predict_into(ctx, local);
			for (std::size_t i = 0; i < probabilities.size(); ++i) {
				probabilities[i] += local[i];
			}
		}
		const auto inv = 1. / static_cast<double>(model.trees.size());
		for (auto& v : probabilities) {
			v *= inv;
		}

		softmax_normalize(probabilities);

		return probabilities;
	}

	uint32_t vocab_size() const {
		return _cfg.vocab_size;
	}

private:
	void softmax_normalize(std::vector<double>& values) const {
		assert(_cfg.softmax_temperature > 0);
		assert(std::isfinite(_cfg.softmax_temperature));

		const auto max_log = *std::max_element(values.begin(), values.end());
		double sum = 0.;
		for (auto& v : values) {
			v = std::exp((v - max_log) / _cfg.softmax_temperature);
			sum += v;
		}
		assert(sum > 0.0);
		assert(std::isfinite(sum));

		for (auto& v : values) {
			v /= sum;
		}
	}

	struct token_model {
		std::vector<event> events;
		std::vector<tree> trees;
	};

	config _cfg;
	std::vector<token_model> _models;
};
}
