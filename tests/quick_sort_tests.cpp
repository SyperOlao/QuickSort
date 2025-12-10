//
// Created by SyperOlao on 11.12.2025.
//

#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include <string>
#include <random>


#include "core/QuickSort.h"
#include "core/Utils.h"
// ----------------- Вспомогательная функция -----------------

template <typename T, typename Compare>
void expect_quicksort_equals_std(std::vector<T> v, Compare cmp) {
    std::vector<T> expected = v;

    QuickSort::sort(v.data(), v.data() + v.size(), cmp);
    std::sort(expected.begin(), expected.end(), cmp);

    ASSERT_EQ(v.size(), expected.size());

    for (size_t i = 0; i < v.size(); ++i) {
        if constexpr (std::is_same_v<T, Point>) {
            // Для Point сравниваем поля
            EXPECT_DOUBLE_EQ(v[i].x, expected[i].x) << "x mismatch at index " << i;
            EXPECT_DOUBLE_EQ(v[i].y, expected[i].y) << "y mismatch at index " << i;
        } else {
            // Для всех остальных типов обычное сравнение
            EXPECT_EQ(v[i], expected[i]) << "mismatch at index " << i;
        }
    }
}

struct PointLess {
    bool operator()(const Point& a, const Point& b) const {
        if (a.x < b.x) return true;
        if (a.x > b.x) return false;
        return a.y < b.y;
    }
};

// ----------------- Тесты для INT -----------------

TEST(QuickSort_Int, EmptyArray) {
    std::vector<int> v;
    QuickSort::sort(v.data(), v.data(), std::less<int>{});
    EXPECT_TRUE(std::is_sorted(v.begin(), v.end()));
}

TEST(QuickSort_Int, SingleElement) {
    std::vector<int> v{42};
    expect_quicksort_equals_std(v, std::less<int>{});
}

TEST(QuickSort_Int, AlreadySorted) {
    std::vector<int> v{-5, -1, 0, 3, 10, 10, 42};
    expect_quicksort_equals_std(v, std::less<int>{});
}

TEST(QuickSort_Int, ReverseSorted) {
    std::vector<int> v{42, 10, 10, 3, 0, -1, -5};
    expect_quicksort_equals_std(v, std::less<int>{});
}

TEST(QuickSort_Int, RandomWithDuplicates) {
    std::vector<int> v{3, 1, 2, 3, 3, -1, 0, 2, 1};
    expect_quicksort_equals_std(v, std::less<int>{});
}

TEST(QuickSort_Int, LargeRandom) {
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(-1000, 1000);
    std::vector<int> v(2000);
    for (auto& x : v) x = dist(rng);

    expect_quicksort_equals_std(v, std::less<int>{});
}

// ----------------- Тесты для DOUBLE -----------------

TEST(QuickSort_Double, BasicCases) {
    std::vector<double> v{3.5, -1.2, 0.0, 3.5, 2.1};
    expect_quicksort_equals_std(v, std::less<double>{});
}

TEST(QuickSort_Double, ReverseSorted) {
    std::vector<double> v{5.0, 4.0, 3.0, 2.0, 1.0};
    expect_quicksort_equals_std(v, std::less<double>{});
}

TEST(QuickSort_Double, LargeRandom) {
    std::mt19937 rng(777);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    std::vector<double> v(1500);
    for (auto& x : v) x = dist(rng);

    expect_quicksort_equals_std(v, std::less<double>{});
}

// ----------------- Тесты для CHAR -----------------

TEST(QuickSort_Char, SimpleString) {
    std::vector<char> v{'z', 'a', 'b', 'a', 'x'};
    expect_quicksort_equals_std(v, std::less<char>{});
}

TEST(QuickSort_Char, LargeRandom) {
    std::mt19937 rng(999);
    std::uniform_int_distribution<int> dist(0, 25);

    std::vector<char> v(1000);
    for (auto& c : v) {
        c = static_cast<char>('a' + dist(rng));
    }

    expect_quicksort_equals_std(v, std::less<char>{});
}

// ----------------- Нетривиальный тип Point -----------------

TEST(QuickSort_Point, SmallManual) {
    std::vector<Point> v{
        {1.0, 2.0},
        {0.0, 10.0},
        {1.0, 1.0},
        {-5.0, 0.0},
        {1.0, 2.0},
    };

    expect_quicksort_equals_std(v, PointLess{});
}

TEST(QuickSort_Point, RandomPoints) {
    std::mt19937 rng(321);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);

    std::vector<Point> v(500);
    for (auto& p : v) {
        p.x = dist(rng);
        p.y = dist(rng);
    }

    expect_quicksort_equals_std(v, PointLess{});
}