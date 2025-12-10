cmake_minimum_required(VERSION 4.0)
project(QuickSort)

set(CMAKE_CXX_STANDARD 26)

add_executable(QuickSort main.cpp
        core/QuickSort.tpp
        core/QuickSort.h
        core/Utils.h
)
