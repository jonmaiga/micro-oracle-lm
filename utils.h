#pragma once

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

	for (const auto& token_trees : forest.trees) {
		size.trees += token_trees.size() * sizeof(oracle_tree);
		for (const auto& tree : token_trees) {
			size.nodes += tree.nodes.size() * sizeof(oracle_node);
			size.leaves += tree.leaves.size() * sizeof(oracle_leaf);

			for (const auto& node : tree.nodes) {
				size.center_contexts += node.center_context.size() * sizeof(feature_id);
			}

			for (const auto& leaf : tree.leaves) {
				size.target_stats += leaf.target_stats.size() * sizeof(target_stat);
				add_unordered_map_storage_bytes(leaf._feature_map, size.feature_map_buckets, size.feature_map_entries);
				for (const auto& [_, target_counts] : leaf._feature_map) {
					add_unordered_map_storage_bytes(target_counts, size.target_count_buckets, size.target_count_entries);
				}
			}
		}
	}

	return size;
}

inline void print_size_bytes(std::ostream& out, const model_size_bytes& size) {
	out << "Model size bytes:\n"
		<< "  total: " << size.total() << '\n'
		<< "  forest: " << size.forest << '\n'
		<< "  token_tree_vectors: " << size.token_tree_vectors << '\n'
		<< "  trees: " << size.trees << '\n'
		<< "  nodes: " << size.nodes << '\n'
		<< "  center_contexts: " << size.center_contexts << '\n'
		<< "  leaves: " << size.leaves << '\n'
		<< "  target_stats: " << size.target_stats << '\n'
		<< "  feature_map_buckets: " << size.feature_map_buckets << '\n'
		<< "  feature_map_entries: " << size.feature_map_entries << '\n'
		<< "  target_count_buckets: " << size.target_count_buckets << '\n'
		<< "  target_count_entries: " << size.target_count_entries << '\n';
}

inline void print_size_bytes(std::ostream& out, const oracle_forest& forest) {
	print_size_bytes(out, size_bytes(forest));
}
}
