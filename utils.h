#pragma once

#include <utility>
#include <unordered_map>

#include "micro_oracle_lm.h"

namespace of {
// Computes bits-per-byte (the average negative base-2 log-likelihood the model
// assigns to each actual next token) over the held-out tokens. Each token maps
// to one byte, so cross-entropy per token equals bits per byte.
double compute_bpb(const of::oracle_forest& forest, const std::vector<token_id>& tokens) {
	constexpr double epsilon = 1e-12;
	const std::int64_t count = tokens.size() > 1 ? static_cast<std::int64_t>(tokens.size()) - 1 : 0;

	// predict() is a read-only, allocation-local operation, so positions can be
	// scored independently and summed via a reduction (matches the OpenMP
	// parallelism already used during training).
	double total_bits = 0.;
#pragma omp parallel for schedule(dynamic, 512) reduction(+ : total_bits)
	for (std::int64_t i = 0; i < count; ++i) {
		const auto probabilities = predict(forest, tokens, static_cast<uint32_t>(i), 1.0);
		const token_id actual = tokens[i + 1];
		assert(actual < probabilities.size());
		const double p = probabilities[actual];
		total_bits += -std::log2(std::max(p, epsilon));
	}
	return count == 0 ? 0. : total_bits / static_cast<double>(count);
}

// Trains a fresh model on the first 90% of the sample and reports bits-per-byte
// on the remaining held-out portion, capped to 10 MiB of test data.
void evaluate_held_out_bpb(const of::oracle_forest_config& base_cfg, const std::vector<token_id>& sample) {
	if (sample.size() < 2) {
		std::cerr << "Sample too small to evaluate held-out BPB.\n";
		return;
	}

	constexpr std::size_t max_test_bytes = 1024ull * 1024ull;
	std::size_t split = sample.size() * 9 / 10;
	if (sample.size() - split > max_test_bytes) {
		split = sample.size() - max_test_bytes;
	}
	if (split >= sample.size()) {
		std::cerr << "Sample too small to create a held-out test split.\n";
		return;
	}
	const std::vector<token_id> train_tokens(sample.begin(), sample.begin() + split);
	const std::vector<token_id> test_tokens(sample.begin() + split, sample.end());
	std::cout << "Held-out evaluation: training on " << train_tokens.size()
		<< " bytes, testing on " << test_tokens.size() << " bytes.\n";

	auto train_start = std::chrono::steady_clock::now();
	const auto forest = build_oracle_forest(base_cfg, {train_tokens});
	auto train_end = std::chrono::steady_clock::now();
	const auto train_seconds = std::chrono::duration<double>(train_end - train_start).count();

	auto eval_start = std::chrono::steady_clock::now();
	const double bpb = compute_bpb(forest, test_tokens);
	auto eval_end = std::chrono::steady_clock::now();
	const auto eval_seconds = std::chrono::duration<double>(eval_end - eval_start).count();

	std::cout << "Held-out BPB: " << bpb << " bits/byte (train " << train_seconds
		<< " s, eval " << eval_seconds << " s).\n";
}

struct model_size_bytes {
	std::size_t forest{};
	std::size_t token_tree_vectors{};
	std::size_t trees{};
	std::size_t nodes{};
	std::size_t center_contexts{};
	std::size_t leaves{};
	std::size_t target_stats{};
	std::size_t feature_keys{};
	std::size_t feature_offsets{};
	std::size_t entries{};
	std::size_t serialized{};

	std::size_t total() const {
		return forest + token_tree_vectors + trees + nodes + center_contexts + leaves + target_stats
			+ feature_keys + feature_offsets + entries;
	}
};

inline model_size_bytes size_bytes(const oracle_forest& forest) {
	model_size_bytes size;
	size.forest += sizeof(forest);
	size.token_tree_vectors += forest.trees.size() * sizeof(std::vector<oracle_tree>);
	size.serialized += sizeof(oracle_forest_config) + sizeof(uint32_t); // cfg + token tree vector count

	for (const auto& token_trees : forest.trees) {
		size.trees += token_trees.size() * sizeof(oracle_tree);
		size.serialized += sizeof(uint32_t); // trees for this token
		for (const auto& tree : token_trees) {
			size.nodes += tree.nodes.size() * sizeof(oracle_node);
			size.leaves += tree.leaves.size() * sizeof(oracle_leaf);
			size.serialized += 2 * sizeof(uint32_t); // node count + leaf count

			for (const auto& node : tree.nodes) {
				size.center_contexts += node.center_context.size() * sizeof(feature_id);
				size.serialized += sizeof(uint32_t); // center context size
				size.serialized += node.center_context.size() * sizeof(feature_id);
				size.serialized += sizeof(double) + 3 * sizeof(uint32_t); // radius + inner + outer + leaf_index
			}

			for (const auto& leaf : tree.leaves) {
				size.target_stats += leaf.target_stats.size() * sizeof(target_stat);
				size.feature_keys += leaf.feature_keys.size() * sizeof(feature_id);
				size.feature_offsets += leaf.feature_offsets.size() * sizeof(uint32_t);
				size.entries += leaf.entry_targets.size() * sizeof(token_id);
				size.entries += leaf.entry_counts.size() * sizeof(uint32_t);
				size.serialized += sizeof(uint32_t); // sample_count
				size.serialized += sizeof(uint32_t); // target stat count
				size.serialized += leaf.target_stats.size() * sizeof(target_stat);
				size.serialized += sizeof(uint32_t); // feature key count
				size.serialized += leaf.feature_keys.size() * sizeof(feature_id);
				size.serialized += leaf.entry_targets.size() * (sizeof(token_id) + sizeof(uint32_t));
			}
		}
	}

	return size;
}

inline void print_size_bytes(std::ostream& out, const model_size_bytes& size) {
	const auto fmt = [](std::size_t bytes) {
		const char* unit = "B";
		double scale = 1.0;
		if (bytes >= 1024ull * 1024ull) {
			unit = "MB";
			scale = 1024.0 * 1024.0;
		}
		else if (bytes >= 1024ull) {
			unit = "KB";
			scale = 1024.0;
		}
		return std::pair{static_cast<double>(bytes) / scale, unit};
	};

	const auto [total_value, total_unit] = fmt(size.total());
	out << "Model size: " << total_value << ' ' << total_unit << "\n";
	const auto print = [&](const char* name, std::size_t bytes) {
		const auto [value, unit] = fmt(bytes);
		out << "  " << name << ": " << value << ' ' << unit << '\n';
	};
	print("forest", size.forest);
	print("token_tree_vectors", size.token_tree_vectors);
	print("trees", size.trees);
	print("nodes", size.nodes);
	print("center_contexts", size.center_contexts);
	print("leaves", size.leaves);
	print("target_stats", size.target_stats);
	print("feature_keys", size.feature_keys);
	print("feature_offsets", size.feature_offsets);
	print("entries", size.entries);
	print("serialized", size.serialized);
}

inline void print_size_bytes(std::ostream& out, const oracle_forest& forest) {
	print_size_bytes(out, size_bytes(forest));
}
}
