cmake_minimum_required(VERSION 4.0)
project(QuickSort)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# -------------------------
# header-only библиотека с QuickSort
# -------------------------
add_library(quicksort INTERFACE)

target_include_directories(quicksort INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# -------------------------
# Основное приложение
# -------------------------
add_executable(QuickSortApp
        main.cpp
)

target_link_libraries(QuickSortApp PRIVATE quicksort)

# -------------------------
# GoogleTest
# -------------------------
include(FetchContent)

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.0.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# -------------------------
# Тесты
# -------------------------
add_executable(QuickSortTests
        tests/quick_sort_tests.cpp
)

target_link_libraries(QuickSortTests PRIVATE
        quicksort
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(QuickSortTests)