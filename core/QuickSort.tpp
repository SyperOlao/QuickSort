//
// Created by SyperOlao on 10.12.2025.
//


#pragma once

#include "QuickSort.h"
#include <utility>


template<typename T>
bool QuickSort::comp(const T &a, const T &b) {
    return a < b;
}

template<typename T, typename Compare>
T &QuickSort::median_of_three(T *first, T *last, Compare comp) {
    T *a = first;
    T *b = first + (last - first) / 2;
    T *c = last - 1;

    if (comp(*b, *a)) std::swap(a, b);
    if (comp(*c, *b)) std::swap(b, c);
    if (comp(*b, *a)) std::swap(a, b);

    return *b;
}

template<typename T, typename Compare>
void QuickSort::insertion_sort(T *first, T *last, Compare comp) {
    for (T *i = first + 1; i < last; ++i) {
        T key = std::move(*i);
        T *j = i;
        while (j > first && comp(key, *(j - 1))) {
            *j = std::move(*(j - 1));
            --j;
        }
        *j = std::move(key);
    }
}


template<typename T, typename Compare>
void QuickSort::sort(T *first, T *last, Compare comp) {
    while (last - first > INSERTION_THRESHOLD) {
        T &pivot_ref = median_of_three(first, last, comp);
        T pivot = pivot_ref;

        T *i = first;
        T *j = last - 1;
        while (true) {
            while (comp(*i, pivot)) {
                ++i;
            }
            while (comp(pivot, *j)) {
                --j;
            }
            if (i >= j) {
                break;
            }
            std::swap(*i, *j);
            ++i;
            --j;
        }

        T *mid = j + 1;

        if (mid - first < last - mid) {
            sort(first, mid, comp);
            first = mid;
        } else {
            sort(mid, last, comp);
            last = mid;
        }
    }

    if (last - first > 1) {
        insertion_sort(first, last, comp);
    }
}
