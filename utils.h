#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "micro_oracle_lm.h"
#include "micro_oracle_tokenizer.h"

namespace of {namespace detail {
	struct byte_count {
		double value;
		const char* unit;
	};

	inline byte_count format_byte_count(std::size_t bytes) {
		constexpr std::size_t kib = 1024;
		constexpr std::size_t mib = kib * kib;
		if (bytes >= mib) {
			return {static_cast<double>(bytes) / static_cast<double>(mib), "MB"};
		}
		if (bytes >= kib) {
			return {static_cast<double>(bytes) / static_cast<double>(kib), "KB"};
		}
		return {static_cast<double>(bytes), "B"};
	}

	inline double seconds_between(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
		return std::chrono::duration<double>(end - start).count();
	}

	inline std::vector<std::vector<token_id>> encode_lines(vocabulary& vocab, const std::vector<unsigned char>& sample) {
		std::vector<std::vector<token_id>> lines;
		std::vector<token_id> line;
		for (const unsigned char byte : sample) {
			line.push_back(vocab.encode(byte));
			if (byte == '\n') {
				lines.push_back(std::move(line));
				line.clear();
			}
		}
		if (!line.empty()) {
			lines.push_back(std::move(line));
		}
		return lines;
	}
	} // namespace detail

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
			return forest + token_tree_vectors + trees + nodes + center_contexts + leaves
				+ target_stats + feature_keys + feature_offsets + entries;
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
		struct byte_count {
			double value;
			const char* unit;
		};

		const auto format = [](std::size_t bytes) -> byte_count {
			if (bytes >= 1024ull * 1024ull) {
				return {static_cast<double>(bytes) / (1024.0 * 1024.0), "MB"};
			}
			if (bytes >= 1024ull) {
				return {static_cast<double>(bytes) / 1024.0, "KB"};
			}
			return {static_cast<double>(bytes), "B"};
		};

		const auto [total_value, total_unit] = format(size.total());
		out << "Model size: " << total_value << ' ' << total_unit << "\n";
		const auto print = [&](const char* name, std::size_t bytes) {
			const auto [value, unit] = format(bytes);
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

	// Computes bits-per-byte over the held-out tokens.
	inline double compute_bpb(const oracle_forest& forest, const std::vector<token_id>& tokens) {
		constexpr double epsilon = 1e-12;
		if (tokens.size() < 2) {
			return 0.;
		}

		const std::int64_t count = static_cast<std::int64_t>(tokens.size() - 1);
		double total_bits = 0.;
#pragma omp parallel for schedule(dynamic, 512) reduction(+ : total_bits)
		for (std::int64_t i = 0; i < count; ++i) {
			const auto probabilities = predict(forest, tokens, static_cast<uint32_t>(i), 1.0);
			const token_id actual = tokens[i + 1];
			assert(actual < probabilities.size());
			const double p = probabilities[actual];
			total_bits += -std::log2(std::max(p, epsilon));
		}
		return total_bits / static_cast<double>(count);
	}

	inline std::string generate(
		const oracle_forest& forest,
		const vocabulary& vocab,
		mx3random& rng,
		int num_tokens,
		double softmax_temperature) {
		std::vector<token_id> tokens{0};
		std::string text;
		const std::size_t token_count = static_cast<std::size_t>(std::max(num_tokens, 0));
		if (token_count > 0) {
			tokens.reserve(token_count + 1);
			text.reserve(token_count);
		}

		for (std::size_t i = 0; i < token_count; ++i) {
			const auto probabilities = predict(forest, tokens, static_cast<uint32_t>(tokens.size() - 1), softmax_temperature);
			assert(!probabilities.empty());

			std::discrete_distribution<uint32_t> token_dist(probabilities.begin(), probabilities.end());
			const token_id next = token_dist(rng);
			text.push_back(vocab.decode(next));
			tokens.push_back(next);
		}
		return text;
	}


	inline void test_tokenizer(oracle_tokenizer_config cfg, oracle_forest_config forest_cfg, const std::vector<unsigned char>& sample) {
		vocabulary vocab;
		const auto samples = detail::encode_lines(vocab, sample);

		forest_cfg.vocab_size = vocab.size();
		std::cout << "Training boundary model on " << samples.size() << " samples...\n";
		const auto model = build_oracle_forest(forest_cfg, samples);

		std::cout << "Training tokenizer...\n";
		const auto train_start = std::chrono::steady_clock::now();
		const auto tokenizer = build_oracle_tokenizer(cfg, model, samples);
		const auto train_seconds = detail::seconds_between(train_start, std::chrono::steady_clock::now());
		std::cout << "Tokenizer training time: " << train_seconds << " s\n";
		std::cout << "Tokenizer vocab size: " << tokenizer.tokens.size() << ".\n";

		const std::size_t base_token_count = vocab.size();
		const std::size_t subunit_count = std::min(tokenizer.tokens.size(), base_token_count + 20);
		std::cout << "Top 20 subunits:\n";
		for (std::size_t i = base_token_count; i < subunit_count; ++i) {
			std::cout << "  " << (i - base_token_count + 1) << ": \"";
			for (const auto token : tokenizer.tokens[i]) {
				std::cout << vocab.decode(token);
			}
			std::cout << "\"\n";
		}
		std::cout << '\n';
	}


	inline void evaluate_held_out_bpb(oracle_forest_config cfg, const std::vector<unsigned char>& sample) {
		assert(sample.size() >= 2);

		constexpr std::size_t max_test_bytes = 1024ull * 1024ull;
		std::size_t split = sample.size() * 9 / 10;
		if (sample.size() - split > max_test_bytes) {
			split = sample.size() - max_test_bytes;
		}
		assert(split < sample.size());

		vocabulary vocab;
		std::vector<token_id> train_tokens;
		train_tokens.reserve(split);
		for (std::size_t i = 0; i < split; ++i) {
			train_tokens.push_back(vocab.encode(sample[i]));
		}

		std::vector<token_id> test_tokens;
		test_tokens.reserve(sample.size() - split);
		for (std::size_t i = split; i < sample.size(); ++i) {
			test_tokens.push_back(vocab.lookup(sample[i]));
		}
		cfg.vocab_size = vocab.size();

		std::cout << "Held-out evaluation: training on " << train_tokens.size() << " bytes, testing on " << test_tokens.size() << " bytes.\n";

		const auto train_start = std::chrono::steady_clock::now();
		const auto forest = build_oracle_forest(cfg, {train_tokens});
		const auto train_seconds = detail::seconds_between(train_start, std::chrono::steady_clock::now());

		const auto eval_start = std::chrono::steady_clock::now();
		const double bpb = compute_bpb(forest, test_tokens);
		const auto eval_seconds = detail::seconds_between(eval_start, std::chrono::steady_clock::now());

		std::cout << "Held-out BPB: " << bpb << " bits/byte (train " << train_seconds << " s, eval " << eval_seconds << " s).\n";

		const auto size_stats = size_bytes(forest);
		print_size_bytes(std::cout, size_stats);
		std::cout << "Model size ratio: " << static_cast<double>(size_stats.serialized) / sample.size() << "\n\n";

		std::cout << "Generating...\n";
		mx3random rng(42);
		const auto generate_start = std::chrono::steady_clock::now();
		std::cout << generate(forest, vocab, rng, 1000, 1.) << '\n';
		const auto generate_seconds = detail::seconds_between(generate_start, std::chrono::steady_clock::now());
		std::cout << "Generation time: " << generate_seconds << " s\n";
	}
}
