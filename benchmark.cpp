//
// Created by SyperOlao on 26.12.2025.
//
#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <cstdio>

#include "core/QuickSort.h"

static void fill_reverse(std::vector<int>& a) {
    const int n = (int)a.size();
    for (int i = 0; i < n; ++i) a[i] = n - i;
}

static std::vector<int> make_median3_killer(size_t n) {
    std::vector<int> a;
    a.reserve(n);
    for (int i = 1; i <= (int)n; i += 2) a.push_back(i);
    for (int i = (int)n - ((n % 2) ? 1 : 0); i >= 2; i -= 2) a.push_back(i);
    for (size_t i = 0; i + 3 < n; i += 4) std::swap(a[i + 1], a[i + 2]);
    return a;
}

static void insertion_sort(std::vector<int>& a) {
    for (size_t i = 1; i < a.size(); ++i) {
        int key = a[i];
        size_t j = i;
        while (j > 0 && key < a[j - 1]) {
            a[j] = a[j - 1];
            --j;
        }
        a[j] = key;
    }
}

static double time_insertion_worst(size_t n, size_t repeats) {
    std::vector<int> base(n);
    fill_reverse(base);

    double total_sec = 0.0;
    for (size_t r = 0; r < repeats; ++r) {
        auto a = base;
        auto t0 = std::chrono::high_resolution_clock::now();
        insertion_sort(a);
        auto t1 = std::chrono::high_resolution_clock::now();
        total_sec += std::chrono::duration<double>(t1 - t0).count();
    }
    return total_sec;
}

static double time_quick(size_t n, size_t repeats, const std::vector<int>& base, std::ptrdiff_t threshold) {
    double total_sec = 0.0;
    for (size_t r = 0; r < repeats; ++r) {
        auto a = base;
        auto t0 = std::chrono::high_resolution_clock::now();
        QuickSort::sort(a.data(), a.data() + (ptrdiff_t)a.size(),
                       [](const int& x, const int& y){ return x < y; },
                       threshold);
        auto t1 = std::chrono::high_resolution_clock::now();
        total_sec += std::chrono::duration<double>(t1 - t0).count();
    }
    return total_sec;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " MinN MaxN Repeats\n";
        return 1;
    }
    size_t minN = std::stoull(argv[1]);
    size_t maxN = std::stoull(argv[2]);
    size_t repeats = std::stoull(argv[3]);

    std::cout
        << "N,"
        << "insertion_worst_sec,"
        << "quick_worst_reverse_sec,"
        << "quick_worst_killer_sec,"
        << "hybrid_worst_killer_sec\n";

    for (size_t n = minN; n <= maxN; ++n) {
        std::cerr << "\n===== N = " << n << " =====\n" << std::flush;
        std::vector<int> rev(n);
        fill_reverse(rev);
        std::vector<int> killer = make_median3_killer(n);

        double ti = time_insertion_worst(n, repeats);

        double tq_rev = time_quick(n, repeats, rev, 1);

        double tq_kill = time_quick(n, repeats, killer, 1);
        double th_kill = time_quick(n, repeats, killer, 16);

        std::cout << n << ","
                  << ti << ","
                  << tq_rev << ","
                  << tq_kill << ","
                  << th_kill << "\n";
    }
    return 0;
}