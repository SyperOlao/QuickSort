//
// Created by SyperOlao on 10.12.2025.
//
#include <random>
#include <type_traits>
#include "QuickSort.h"

template<typename T>
void fill_random(T* first, const T* last) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<long long> dist(
            std::numeric_limits<T>::min(),
            std::numeric_limits<T>::max()
        );

        for (T* it = first; it != last; ++it) {
            *it = static_cast<T>(dist(gen));
        }

    } else if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(
            static_cast<T>(-100000.0),
            static_cast<T>(100000.0)
        );

        for (T* it = first; it != last; ++it) {
            *it = dist(gen);
        }

    } else {
        static_assert(std::is_arithmetic_v<T>,
            "fill_random supports only arithmetic types");
    }
}

template<>
void fill_random<char>(char* first, const char* last) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist('a', 'z');

    for (char* it = first; it != last; ++it) {
        *it = static_cast<char>(dist(gen));
    }
}

template<typename T>
void run(std::size_t n) {
    std::vector<T> data(n);

    for (std::size_t i = 0; i < n; ++i) {
        std::cin >> data[i];
    }

    const auto start = std::chrono::high_resolution_clock::now();

    QuickSort::sort(
        data.data(),
        data.data() + n,
        [](const T& a, const T& b) { return a < b; }
    );

    const auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> t = end - start;


    std::cout << "TIME_MS " << t.count() << "\n";
    std::cout << "DATA";

    for (const auto& x : data) {
        std::cout << " " << x;
    }

    std::cout << "\n";
}
