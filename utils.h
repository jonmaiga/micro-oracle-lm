#pragma once

#include <ostream>
#include <utility>
#include <unordered_map>

#include "micro_oracle_lm.h"

namespace of {
struct model_size_bytes {
	std::size_t forest{};
	std::size_t token_tree_vectors{};
	std::size_t trees{};
	std::size_t nodes{};
	std::size_t center_contexts{};
	std::size_t leaves{};
	std::size_t target_stats{};
	std::size_t feature_map_buckets{};
	std::size_t feature_map_entries{};
	std::size_t target_count_buckets{};
	std::size_t target_count_entries{};
	std::size_t serialized{};

	std::size_t total() const {
		return forest + token_tree_vectors + trees + nodes + center_contexts + leaves + target_stats
			+ feature_map_buckets + feature_map_entries + target_count_buckets + target_count_entries;
	}
};

template <class K, class V>
void add_unordered_map_storage_bytes(const std::unordered_map<K, V>& map,
                                     std::size_t& buckets, std::size_t& entries) {
	buckets += map.bucket_count() * sizeof(void*);
	entries += map.size() * sizeof(typename std::unordered_map<K, V>::value_type);
}

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
				size.serialized += sizeof(uint32_t); // sample_count
				size.serialized += sizeof(uint32_t); // target stat count
				size.serialized += leaf.target_stats.size() * sizeof(target_stat);
				add_unordered_map_storage_bytes(leaf._feature_map, size.feature_map_buckets, size.feature_map_entries);
				size.serialized += sizeof(uint32_t); // feature map entry count
				for (const auto& [_, target_counts] : leaf._feature_map) {
					add_unordered_map_storage_bytes(target_counts, size.target_count_buckets, size.target_count_entries);
					size.serialized += sizeof(feature_id); // feature key
					size.serialized += sizeof(uint32_t); // target-count entry count
					size.serialized += target_counts.size() * (sizeof(token_id) + sizeof(uint32_t));
				}
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
	print("feature_map_buckets", size.feature_map_buckets);
	print("feature_map_entries", size.feature_map_entries);
	print("target_count_buckets", size.target_count_buckets);
	print("target_count_entries", size.target_count_entries);
	print("serialized", size.serialized);
}

inline void print_size_bytes(std::ostream& out, const oracle_forest& forest) {
	print_size_bytes(out, size_bytes(forest));
}
}
