#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#if defined(__aarch64__) || defined(__arm64__)
#include <arm_neon.h>
#define USE_NEON 1
#endif

#if defined(__clang__)
#pragma float_control(precise, on)
#endif

namespace {

constexpr uint32_t ITERATIONS = 200000000;
constexpr double SCALE = 9007199254740992.0;

inline int64_t nearest_integer(double value) {
#ifdef USE_NEON
    return vcvtnd_s64_f64(value);
#else
    return static_cast<int64_t>(std::nearbyint(value));
#endif
}

#ifdef USE_NEON
inline int64x2_t pair_difference(float64x2_t denominator,
                                 float64x2_t scale,
                                 float64x2_t two) {
    const int64x2_t minus =
        vcvtnq_s64_f64(vdivq_f64(scale, denominator));
    const int64x2_t plus =
        vcvtnq_s64_f64(vdivq_f64(scale, vaddq_f64(denominator, two)));
    return vsubq_s64(plus, minus);
}
#endif

int64_t ordinary_sum(uint32_t first, uint32_t end) {
    int64_t sum = 0;

#ifdef USE_NEON
    if (end - first >= 8) {
        const float64x2_t scale = vdupq_n_f64(SCALE);
        const float64x2_t two = vdupq_n_f64(2.0);
        const float64x2_t step = vdupq_n_f64(32.0);

        const double base = 4.0 * first - 1.0;
        float64x2_t d0 = {base, base + 4.0};
        float64x2_t d1 = vaddq_f64(d0, vdupq_n_f64(8.0));
        float64x2_t d2 = vaddq_f64(d0, vdupq_n_f64(16.0));
        float64x2_t d3 = vaddq_f64(d0, vdupq_n_f64(24.0));

        int64x2_t a0 = vdupq_n_s64(0);
        int64x2_t a1 = vdupq_n_s64(0);
        int64x2_t a2 = vdupq_n_s64(0);
        int64x2_t a3 = vdupq_n_s64(0);

        do {
            a0 = vaddq_s64(a0, pair_difference(d0, scale, two));
            a1 = vaddq_s64(a1, pair_difference(d1, scale, two));
            a2 = vaddq_s64(a2, pair_difference(d2, scale, two));
            a3 = vaddq_s64(a3, pair_difference(d3, scale, two));

            d0 = vaddq_f64(d0, step);
            d1 = vaddq_f64(d1, step);
            d2 = vaddq_f64(d2, step);
            d3 = vaddq_f64(d3, step);
            first += 8;
        } while (end - first >= 8);

        sum = vaddvq_s64(
            vaddq_s64(vaddq_s64(a0, a1), vaddq_s64(a2, a3)));
    }
#endif

    for (; first < end; ++first) {
        const double denominator = 4.0 * first;
        sum -= nearest_integer(SCALE / (denominator - 1.0));
        sum += nearest_integer(SCALE / (denominator + 1.0));
    }

    return sum;
}

inline void apply_term(int64_t& result, uint32_t denominator, int sign) {
    const double value = SCALE / static_cast<double>(denominator);
    const int64_t rounded = nearest_integer(value);

    result += sign * rounded;

    if (result & 1) {
        const double remainder = value - static_cast<double>(rounded);
        if (remainder == 0.5)
            result += sign;
        else if (remainder == -0.5)
            result -= sign;
    }
}

template <std::size_t N>
void add_midpoint_iterations(const uint64_t (&factors)[N],
                             std::vector<uint32_t>& indices) {
    std::vector<uint64_t> divisors;
    divisors.reserve(std::size_t{1} << N);
    divisors.push_back(1);

    for (uint64_t factor : factors) {
        const std::size_t count = divisors.size();
        for (std::size_t i = 0; i < count; ++i)
            divisors.push_back(divisors[i] * factor);
    }

    constexpr uint64_t maximum = 4ULL * ITERATIONS + 1;
    for (uint64_t divisor : divisors) {
        if (divisor < 3 || divisor > maximum)
            continue;

        const uint64_t index =
            (divisor & 3) == 3 ? (divisor + 1) / 4 : (divisor - 1) / 4;

        if (index >= 1 && index <= ITERATIONS)
            indices.push_back(static_cast<uint32_t>(index));
    }
}

double calculate() {
    // Partial sums occupy the 2^-53 lattice. Handle halfway cases separately.
    // Their denominators divide 2^54 - 1 or 2^54 + 1.
    constexpr uint64_t minus_factors[] = {
        3, 3, 3, 3, 7, 19, 73, 87211, 262657
    };
    constexpr uint64_t plus_factors[] = {
        5, 13, 37, 109, 246241, 279073
    };

    std::vector<uint32_t> midpoints;
    midpoints.reserve(576);
    add_midpoint_iterations(minus_factors, midpoints);
    add_midpoint_iterations(plus_factors, midpoints);

    std::sort(midpoints.begin(), midpoints.end());
    midpoints.erase(std::unique(midpoints.begin(), midpoints.end()),
                    midpoints.end());

    int64_t result = int64_t{1} << 53;
    uint32_t first = 1;

    for (uint32_t index : midpoints) {
        result += ordinary_sum(first, index);
        apply_term(result, 4 * index - 1, -1);
        apply_term(result, 4 * index + 1, 1);
        first = index + 1;
    }

    result += ordinary_sum(first, ITERATIONS + 1);
    return static_cast<double>(result) * 0x1p-51;
}

} // namespace

int main() {
    const auto start = std::chrono::steady_clock::now();
    const double result = calculate();
    const auto end = std::chrono::steady_clock::now();

    const double elapsed = std::chrono::duration<double>(end - start).count();
    std::printf("Result: %.12f\nExecution Time: %.6f seconds\n",
                result, elapsed);
    return 0;
}