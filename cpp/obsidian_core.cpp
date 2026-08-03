// Obsidian Terminal - C++ performance kernels
// Built with: clang++ -O3 -std=c++17 -shared -fPIC cpp/obsidian_core.cpp -o libobsidian_core.dylib
// C ABI (ctypes-compatible) - no external dependencies.

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

// Signal encodings (mirror strategies.trend_following.Signal)
enum { SIGNAL_HOLD = 0, SIGNAL_BUY = 1, SIGNAL_STRONG_BUY = 2, SIGNAL_SELL = 3, SIGNAL_STRONG_SELL = 4 };

// Streaming SMA crossover signal series.
// Replicates SMAStrategy.generate_signal exactly, single pass over prices.
// signals[i] / confidences[i] filled for i in [0, n). Returns 0 on success.
int sma_crossover_signals(const double* prices, int n, int short_w, int long_w,
                          double threshold, int* signals, double* confidences) {
    if (n <= 0 || short_w <= 0 || long_w <= 0 || short_w >= long_w) return 1;

    // Prefix sums for O(1) window means.
    std::vector<double> pref(n + 1, 0.0);
    for (int i = 0; i < n; ++i) pref[i + 1] = pref[i] + prices[i];

    auto window_mean = [&](int end, int w) -> double {
        if (end < w) return NAN;
        return (pref[end] - pref[end - w]) / (double)w;
    };

    for (int i = 0; i < n; ++i) {
        // Bars before the long window are not ready -> HOLD (matches Python).
        // When i+1 >= long_w, both iloc[-1] and iloc[-2] are valid because
        // short_w < long_w, so the first real signal is at bar long_w - 1.
        if (i + 1 < long_w) {
            signals[i] = SIGNAL_HOLD;
            confidences[i] = 0.0;
            continue;
        }

        double cur_short = window_mean(i + 1, short_w);
        double cur_long = window_mean(i + 1, long_w);
        double prev_short = window_mean(i, short_w);
        double prev_long = window_mean(i, long_w);

        bool cross_above = prev_short <= prev_long && cur_short > cur_long;
        bool cross_below = prev_short >= prev_long && cur_short < cur_long;

        double distance = (cur_short - cur_long) / cur_long;
        double confidence = fmin(fabs(distance) / 0.05, 1.0);

        int sig;
        if (cross_above) {
            sig = (distance > threshold * 2.0) ? SIGNAL_STRONG_BUY : SIGNAL_BUY;
        } else if (cross_below) {
            sig = (distance < -threshold * 2.0) ? SIGNAL_STRONG_SELL : SIGNAL_SELL;
        } else if (cur_short > cur_long * (1.0 + threshold)) {
            sig = SIGNAL_BUY;
            confidence *= 0.7;
        } else if (cur_short < cur_long * (1.0 - threshold)) {
            sig = SIGNAL_SELL;
            confidence *= 0.7;
        } else {
            sig = SIGNAL_HOLD;
            confidence = 0.5;
        }
        signals[i] = sig;
        confidences[i] = confidence;
    }
    return 0;
}

// Deterministic xorshift64 RNG.
static inline uint64_t xs64(uint64_t* s) {
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    return x;
}

static inline double u01(uint64_t* s) {
    return (double)(xs64(s) >> 11) / 9007199254740992.0; // / 2^53
}

// Bootstrap Monte Carlo simulation.
// For each of n_sim simulations: sample n_days returns with replacement,
// compound equity, track max drawdown. Parallelized when OpenMP available.
// Returns 0 on success.
int monte_carlo(const double* returns, int n_days, int n_sim, double initial_capital,
                uint64_t seed, double* out_finals, double* out_drawdowns) {
    if (n_days <= 0 || n_sim <= 0) return 1;

    std::vector<double> tmp(returns, returns + n_days); // local copy (may be mutated)

#pragma omp parallel for
    for (int sim = 0; sim < n_sim; ++sim) {
        uint64_t s = seed ^ ((uint64_t)sim * 0x9E3779B97F4A7C15ULL);
        double equity = initial_capital;
        double peak = initial_capital;
        double min_dd = 0.0;
        for (int d = 0; d < n_days; ++d) {
            int idx = (int)(u01(&s) * (double)n_days);
            if (idx >= n_days) idx = n_days - 1;
            equity *= (1.0 + tmp[idx]);
            if (equity > peak) peak = equity;
            double dd = (equity - peak) / peak;
            if (dd < min_dd) min_dd = dd;
        }
        out_finals[sim] = equity;
        out_drawdowns[sim] = min_dd;
    }
    return 0;
}

// Streaming max drawdown over an equity curve. Returns 0 on success.
int max_drawdown(const double* equity, int n, double* out_dd) {
    if (n <= 0) return 1;
    double peak = equity[0];
    double min_dd = 0.0;
    for (int i = 1; i < n; ++i) {
        if (equity[i] > peak) peak = equity[i];
        double dd = (equity[i] - peak) / peak;
        if (dd < min_dd) min_dd = dd;
    }
    *out_dd = min_dd;
    return 0;
}

} // extern "C"
