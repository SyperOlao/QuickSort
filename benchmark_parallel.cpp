//
// Created by SyperOlao on 26.12.2025.
//
#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <cstddef>
#include <atomic>
#include <cstring>
#include <fstream>
#include <thread>
#include <cmath>
#include "core/QuickSort.h"

// честно скажу это все с джпт, потому что оно очень долго считало и я решила распараллелить, чтобы побыстрее было

#ifdef _OPENMP
  #include <omp.h>
#endif

using Clock = std::chrono::high_resolution_clock;

static void fill_reverse(std::vector<int>& a) {
    const int n = (int)a.size();
    for (int i = 0; i < n; ++i) a[i] = n - i;
}

// Deterministic "killer-ish" pattern for median-of-three variants
static std::vector<int> make_median3_killer(size_t n) {
    std::vector<int> a;
    a.reserve(n);

    for (int i = 1; i <= (int)n; i += 2) a.push_back(i);
    for (int i = (int)n - ((n % 2) ? 1 : 0); i >= 2; i -= 2) a.push_back(i);

    for (size_t i = 0; i + 3 < n; i += 4) std::swap(a[i + 1], a[i + 2]);
    return a;
}

static void insertion_sort_raw(int* first, int* last) {
    for (int* i = first + 1; i < last; ++i) {
        int key = *i;
        int* j = i;
        while (j > first && key < *(j - 1)) {
            *j = *(j - 1);
            --j;
        }
        *j = key;
    }
}

static double bench_insertion_worst_parallel(
    const std::vector<int>& base,
    long long repeats,
    int threads,
    std::atomic<long long>* done /* may be null */
) {
    double total_sec = 0.0;

#ifdef _OPENMP
    if (threads > 0) {
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
    }

    const size_t n = base.size();

    #pragma omp parallel reduction(+:total_sec)
    {
        std::vector<int> a(n);

        #pragma omp for schedule(static)
        for (long long r = 0; r < repeats; ++r) {
            // memcpy is a tiny bit faster than std::copy for POD ints
            std::memcpy(a.data(), base.data(), n * sizeof(int));

            auto t0 = Clock::now();
            insertion_sort_raw(a.data(), a.data() + (std::ptrdiff_t)n);
            auto t1 = Clock::now();

            total_sec += std::chrono::duration<double>(t1 - t0).count();
            if (done) done->fetch_add(1, std::memory_order_relaxed);
        }
    }
#else
    (void)threads;
    const size_t n = base.size();
    std::vector<int> a(n);

    for (long long r = 0; r < repeats; ++r) {
        std::memcpy(a.data(), base.data(), n * sizeof(int));
        auto t0 = Clock::now();
        insertion_sort_raw(a.data(), a.data() + (std::ptrdiff_t)n);
        auto t1 = Clock::now();
        total_sec += std::chrono::duration<double>(t1 - t0).count();
        if (done) done->fetch_add(1, std::memory_order_relaxed);
    }
#endif

    return total_sec;
}

static double bench_quick_parallel(
    const std::vector<int>& base,
    long long repeats,
    std::ptrdiff_t threshold,
    int threads,
    std::atomic<long long>* done /* may be null */
) {
    double total_sec = 0.0;

#ifdef _OPENMP
    if (threads > 0) {
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
    }

    const size_t n = base.size();

    #pragma omp parallel reduction(+:total_sec)
    {
        std::vector<int> a(n);

        #pragma omp for schedule(static)
        for (long long r = 0; r < repeats; ++r) {
            std::memcpy(a.data(), base.data(), n * sizeof(int));

            auto t0 = Clock::now();
            QuickSort::sort(
                a.data(), a.data() + (std::ptrdiff_t)n,
                [](const int& x, const int& y) { return x < y; },
                threshold
            );
            auto t1 = Clock::now();

            total_sec += std::chrono::duration<double>(t1 - t0).count();
            if (done) done->fetch_add(1, std::memory_order_relaxed);
        }
    }
#else
    (void)threads;
    const size_t n = base.size();
    std::vector<int> a(n);

    for (long long r = 0; r < repeats; ++r) {
        std::memcpy(a.data(), base.data(), n * sizeof(int));
        auto t0 = Clock::now();
        QuickSort::sort(
            a.data(), a.data() + (std::ptrdiff_t)n,
            [](const int& x, const int& y) { return x < y; },
            threshold
        );
        auto t1 = Clock::now();
        total_sec += std::chrono::duration<double>(t1 - t0).count();
        if (done) done->fetch_add(1, std::memory_order_relaxed);
    }
#endif

    return total_sec;
}

static void print_openmp_info(int threads_requested) {
#ifdef _OPENMP
    int maxT = omp_get_max_threads();
    std::cerr << "OpenMP ON, max threads = " << maxT << "\n";
    if (threads_requested > 0) {
        omp_set_dynamic(0);
        omp_set_num_threads(threads_requested);
    }
    #pragma omp parallel
    {
        #pragma omp single
        std::cerr << "OpenMP actual threads = " << omp_get_num_threads() << "\n";
    }
#else
    (void)threads_requested;
    std::cerr << "OpenMP OFF\n";
#endif
}
static std::vector<size_t> make_log_grid(size_t minN, size_t maxN, int points) {
    std::vector<size_t> ns;
    if (minN < 2) minN = 2;
    if (maxN < minN) maxN = minN;

    if (points < 2) points = 2;
    ns.reserve(points);

    const double a = std::log((double)minN);
    const double b = std::log((double)maxN);

    size_t prev = 0;
    for (int i = 0; i < points; ++i) {
        double t = (points == 1) ? 0.0 : (double)i / (double)(points - 1);
        size_t n = (size_t)std::llround(std::exp(a + (b - a) * t));
        if (n < minN) n = minN;
        if (n > maxN) n = maxN;

        if (n != prev) ns.push_back(n);
        prev = n;
    }

    if (ns.empty()) ns.push_back(minN);
    if (ns.back() != maxN) ns.push_back(maxN);
    return ns;
}

int main(int argc, char** argv) {
     // Usage:
    // benchmark_parallel MinN MaxN Repeats [Threads] [--out out.csv]
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " MinN MaxN Repeats [Threads] [--out out.csv]\n";
        return 1;
    }

    size_t minN = std::stoull(argv[1]);
    size_t maxN = std::stoull(argv[2]);
    long long repeats = std::stoll(argv[3]);

    int threads = 0;
    if (argc >= 5)
        threads = std::atoi(argv[4]);

    std::string out_path;
    for (int i = 5; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--out")
            out_path = argv[i + 1];
    }

    print_openmp_info(threads);

    std::ofstream fout;
    std::ostream* out = &std::cout;
    if (!out_path.empty()) {
        fout.open(out_path, std::ios::out | std::ios::trunc);
        if (!fout) {
            std::cerr << "Failed to open output file: " << out_path << "\n";
            return 2;
        }
        out = &fout;
        std::cerr << "Writing CSV to: " << out_path << "\n";
    }

    // CSV header
    (*out) << "N,"
           << "insertion_worst_sec,"
           << "quick_worst_reverse_sec,"
           << "quick_worst_killer_sec,"
           << "hybrid_worst_killer_sec\n";
    out->flush();

    // ---- логарифмическая сетка N ----
    const int POINTS = 25; // оптимально: 20–30 точек
    auto Ns = make_log_grid(minN, maxN, POINTS);

    auto last_tick = Clock::now();

    for (size_t idx = 0; idx < Ns.size(); ++idx) {
        size_t n = Ns[idx];

        std::cerr << "\n=== N = " << n
                  << " (" << (idx + 1) << "/" << Ns.size() << ") ===\n";

        std::vector<int> rev(n);
        fill_reverse(rev);

        std::vector<int> killer = make_median3_killer(n);

        std::atomic<long long> done{0};
        const long long total = repeats * 4;

        auto report_progress = [&]() {
            auto now = Clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now - last_tick)
                          .count();
            if (ms >= 500) {
                last_tick = now;
                long long d = done.load(std::memory_order_relaxed);
                double pct = 100.0 * double(d) / double(total);
                std::cerr << "progress: " << d << "/" << total
                          << " (" << pct << "%)\n";
            }
        };

        auto t0 = Clock::now();

        double ti = bench_insertion_worst_parallel(
            rev, repeats, threads, &done);
        report_progress();

        double tq_rev = bench_quick_parallel(
            rev, repeats, 1, threads, &done);
        report_progress();

        double tq_kill = bench_quick_parallel(
            killer, repeats, 1, threads, &done);
        report_progress();

        double th_kill = bench_quick_parallel(
            killer, repeats, 16, threads, &done);
        report_progress();

        auto t1 = Clock::now();
        double secN =
            std::chrono::duration<double>(t1 - t0).count();

        std::cerr << "N finished in " << secN << " sec\n";

        (*out) << n << ","
               << ti << ","
               << tq_rev << ","
               << tq_kill << ","
               << th_kill << "\n";
        out->flush();
    }

    return 0;

    return 0;
}