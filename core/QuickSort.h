//
// Created by SyperOlao on 10.12.2025.
//

#ifndef QUICKSORT_QUICKSORT_H
#define QUICKSORT_QUICKSORT_H
#include <cstddef>

class QuickSort {
    static constexpr std::ptrdiff_t INSERTION_THRESHOLD = 16;
    template<typename T>
    static bool comp(const T &a, const T &b);

    template<typename T, typename Compare>
    static T& median_of_three(T *first, T *last, Compare comp = QuickSort::comp);
    template<typename T, typename Compare>
    static void insertion_sort(T* first, T* last, Compare comp= QuickSort::comp);

public:
    template<typename T, typename Compare>
    static void sort(T *first, T *last, Compare comp = QuickSort::comp);
};

#include "QuickSort.tpp"
#endif //QUICKSORT_QUICKSORT_H
