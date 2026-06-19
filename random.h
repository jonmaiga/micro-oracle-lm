#pragma once

#include <cstdint>
#include <limits>

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
