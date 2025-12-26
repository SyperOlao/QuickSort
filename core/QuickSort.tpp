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
void QuickSort::sort(T *first, T *last, Compare comp, std::ptrdiff_t insertion_threshold) {
    if (!first || !last) return;
    if (last - first <= 1) return;
    if (insertion_threshold < 1) insertion_threshold = 1;


    while (last - first > insertion_threshold) {
        T pivot = median_of_three(first, last, comp);

        T *lt = first;
        T *i  = first;
        T *gt = last;

        while (i < gt) {
            if (comp(*i, pivot)) {
                std::swap(*lt, *i);
                ++lt;
                ++i;
            } else if (comp(pivot, *i)) {
                --gt;
                std::swap(*i, *gt);
            } else {
                ++i;
            }
        }

        const auto leftSize  = lt - first;
        const auto rightSize = last - gt;

        if (leftSize < rightSize) {
            if (leftSize > 1) sort(first, lt, comp, insertion_threshold);
            first = gt;
        } else {
            if (rightSize > 1) sort(gt, last, comp, insertion_threshold);
            last = lt;
        }
    }

    if (last - first > 1) insertion_sort(first, last, comp);
}


template<typename T, typename Compare>
void QuickSort::sort(T *first, T *last, Compare comp) {
    sort(first, last, comp, INSERTION_THRESHOLD);
}
