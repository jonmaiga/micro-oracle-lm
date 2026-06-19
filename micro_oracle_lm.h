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

namespace micro_oracle {
using token_id = uint32_t;
using feature_id = uint64_t;
using context = std::vector<feature_id>;

static const uint64_t C = 0xbea225f9eb34556d;

inline uint64_t mx3mix(uint64_t x) {
	x ^= x >> 32;
	x *= C;
	x ^= x >> 29;
	x *= C;
	x ^= x >> 32;
	x *= C;
	x ^= x >> 29;
	return x;
}

class mx3random {
public:
	using result_type = uint64_t;

	explicit mx3random(uint64_t seed) : _seed(mx3mix(seed + C)), _counter(mx3mix(seed - C)) {
	}

	uint64_t operator()() {
		_counter ^= _seed;
		_counter += C;
		return mx3mix(_counter);
	}

	static constexpr uint64_t min() { return 0; }
	static constexpr uint64_t max() { return std::numeric_limits<uint64_t>::max(); }

private:
	uint64_t _seed;
	uint64_t _counter;
};

struct config {
	uint32_t vocab_size{};
	uint32_t context_size{6};
	uint32_t ensemble_size{8};
	uint32_t max_depth{8};
	double smoothing{0.5};
	double softmax_temperature{1.};
};

inline feature_id make_feature(token_id token, uint32_t distance, uint32_t vocab_size) {
	return static_cast<feature_id>(token) + static_cast<feature_id>(distance) * vocab_size;
}

inline context make_context_at(const std::vector<token_id>& tokens, uint32_t index,
                               uint32_t context_size, uint32_t vocab_size) {
	context ctx;
	const auto start = index > context_size ? index - context_size : 0;
	const uint32_t count = index - start;
	ctx.reserve(count);
	for (uint32_t distance = 1; distance <= count; ++distance) {
		ctx.push_back(make_feature(tokens[index - distance], distance, vocab_size));
	}
	return ctx;
}

// A training position referenced lazily into its token stream. The context and
// next token are derived on demand instead of materialized per event, keeping
// memory proportional to the corpus rather than corpus * context_size.
struct event {
	const std::vector<token_id>* tokens{};
	uint32_t index{};

	context ctx(uint32_t context_size, uint32_t vocab_size) const {
		return make_context_at(*tokens, index, context_size, vocab_size);
	}

	token_id next() const {
		return (*tokens)[index + 1];
	}
};

inline void softmax_normalize(std::vector<double>& values, double temperature) {
	assert(temperature > 0);
	assert(std::isfinite(temperature));

	const auto max_log = *std::ranges::max_element(values);
	double sum = 0.;
	for (auto& v : values) {
		v = std::exp((v - max_log) / temperature);
		sum += v;
	}
	assert(sum > 0.0);
	assert(std::isfinite(sum));

	for (auto& v : values) {
		v /= sum;
	}
}

// Cosine distance over the (binary) feature sets. Both feature lists are sorted
// by construction, so overlap is a linear merge.
inline double distance(const context& a, const context& b) {
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

class leaf_model {
public:
	explicit leaf_model(uint32_t vocab_size) : _label_stats(vocab_size) {
	}

	void observe(const context& ctx, token_id label) {
		assert(label < _label_stats.size());

		++_sample_count;
		++_label_stats[label].sample_total;
		_label_stats[label].feature_total += static_cast<uint32_t>(ctx.size());

		for (const auto f : ctx) {
			++_feature_map[f][label];
		}
	}

	void predict_into(const context& ctx, std::vector<double>& out, double smoothing) const {
		assert(!_label_stats.empty());
		assert(out.size() == _label_stats.size());
		assert(_sample_count > 0);

		const auto class_count = _label_stats.size();

		const double prior_den = static_cast<double>(_sample_count) + smoothing * static_cast<double>(class_count);
		for (std::size_t label = 0; label < class_count; ++label) {
			const auto count = _label_stats[label].sample_total;
			out[label] += std::log((static_cast<double>(count) + smoothing) / prior_den);
		}

		// todo: replace with global vocab_size?
		const auto vocabulary_size = static_cast<double>(_feature_map.size());
		for (const auto f : ctx) {
			const auto it = _feature_map.find(f);
			if (it == _feature_map.end()) {
				continue;
			}
			const auto& counts = it->second;
			for (token_id label = 0; label < class_count; ++label) {
				const auto total = _label_stats[label].feature_total;
				const auto cit = counts.find(label);
				const auto count = cit != counts.end() ? cit->second : 0;
				const double den = static_cast<double>(total) + smoothing * vocabulary_size;
				out[label] += std::log((static_cast<double>(count) + smoothing) / den);
				assert(std::isfinite(out[label]));
			}
		}
	}

private:
	struct label_stat {
		uint32_t sample_total{};
		uint32_t feature_total{};
	};

	uint32_t _sample_count{};
	std::vector<label_stat> _label_stats;
	// feature -> next_token->count
	std::unordered_map<feature_id, std::unordered_map<token_id, uint32_t>> _feature_map;
};

class tree {
public:
	tree(const config& cfg) :
		_cfg(cfg) {
	}

	void build(std::span<const event> events, std::vector<uint32_t> event_indices, mx3random& rng) {
		_nodes.clear();
		_leaves.clear();
		build_node(events, event_indices, 0, rng);
	}

	void predict_into(const context& ctx, std::vector<double>& out) const {
		const auto node_index = route(ctx);
		_leaves[_nodes[node_index].leaf_index].predict_into(ctx, out, _cfg.smoothing);
	}

private:
	struct node {
		bool leaf{true};
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

	uint32_t build_node(std::span<const event> events, std::span<uint32_t> event_indices, uint32_t depth, mx3random& rng) {
		const auto event_count = event_indices.size();
		if (depth >= _cfg.max_depth || event_count < 2) {
			return build_leaf(events, event_indices);
		}

		std::uniform_int_distribution<std::size_t> dist(0, event_count - 1);
		const auto center_ctx = events[event_indices[dist(rng)]].ctx(_cfg.context_size, _cfg.vocab_size);
		constexpr int radius_samples = 7;
		double radius = 0.;
		for (int i = 0; i < 7; ++i) {
			radius += distance(events[event_indices[dist(rng)]].ctx(_cfg.context_size, _cfg.vocab_size), center_ctx);
		}
		radius /= radius_samples;

		const auto split = std::ranges::stable_partition(event_indices, [&](const auto index) {
			return distance(events[index].ctx(_cfg.context_size, _cfg.vocab_size), center_ctx) < radius;
		});
		if (split.empty() || split.size() == event_count) {
			return build_leaf(events, event_indices);
		}

		const auto mid = static_cast<std::size_t>(split.begin() - event_indices.begin());
		const auto node_index = _nodes.size();
		_nodes.push_back({.leaf = false, .center_context = center_ctx, .radius = radius});
		_nodes[node_index].inner = build_node(events, event_indices.first(mid), depth + 1, rng);
		_nodes[node_index].outer = build_node(events, event_indices.subspan(mid), depth + 1, rng);
		return static_cast<uint32_t>(node_index);
	}

	uint32_t build_leaf(std::span<const event> events, std::span<const uint32_t> source) {
		const auto leaf_index = static_cast<uint32_t>(_leaves.size());
		auto& leaf = _leaves.emplace_back(_cfg.vocab_size);
		for (const auto index : source) {
			const auto& e = events[index];
			leaf.observe(e.ctx(_cfg.context_size, _cfg.vocab_size), e.next());
		}

		const auto node_index = _nodes.size();
		_nodes.push_back({.leaf = true, .leaf_index = leaf_index});
		return static_cast<uint32_t>(node_index);
	}

	config _cfg;
	std::vector<node> _nodes;
	std::vector<leaf_model> _leaves;
};

class micro_oracle_lm {
public:
	explicit micro_oracle_lm(const config& cfg) : _cfg(cfg) {
	}

	void train(const std::vector<std::vector<token_id>>& samples) {
		_models.assign(_cfg.vocab_size, {});

		std::vector<std::vector<event>> events(_cfg.vocab_size);
		for (const auto& sample : samples) {
			for (std::size_t i = 0; i + 1 < sample.size(); ++i) {
				events[sample[i]].push_back({&sample, static_cast<uint32_t>(i)});
			}
		}

		const int token_count = static_cast<int>(_models.size());
#pragma omp parallel for schedule(dynamic, 1)
		for (int token = 0; token < token_count; ++token) {
			auto& token_events = events[token];
			if (token_events.empty()) {
				continue;
			}
			mx3random rng(token);
			std::vector<uint32_t> indices(token_events.size());
			std::iota(indices.begin(), indices.end(), 0u);
			auto& trees = _models[token];
			trees.reserve(_cfg.ensemble_size);
			for (uint32_t t = 0; t < _cfg.ensemble_size; ++t) {
				auto& tr = trees.emplace_back(_cfg);
				tr.build(token_events, indices, rng);
			}
		}
	}

	std::vector<double> predict(const std::vector<token_id>& tokens, uint32_t index_to_predict) const {
		assert(index_to_predict < tokens.size());

		const auto current = tokens[index_to_predict];
		if (current >= _models.size() || _models[current].empty()) {
			// todo: oov
			return std::vector(_cfg.vocab_size, 1. / _cfg.vocab_size);
		}

		const auto ctx = make_context_at(tokens, index_to_predict, _cfg.context_size, _cfg.vocab_size);
		const auto& trees = _models[current];

		std::vector<double> probabilities(_cfg.vocab_size);
		for (const auto& t : trees) {
			t.predict_into(ctx, probabilities);
		}
		const auto inv = 1. / static_cast<double>(trees.size());
		for (auto& v : probabilities) {
			v *= inv;
		}

		softmax_normalize(probabilities, _cfg.softmax_temperature);

		return probabilities;
	}

private:
	config _cfg;
	std::vector<std::vector<tree>> _models;
};
}
