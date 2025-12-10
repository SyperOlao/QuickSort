#include <iostream>
#include <chrono>
#include "core/Utils.h"
#include "core/QuickSort.h"

// TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.


int main(int argc, char** argv) {
    if (argc < 3) return 1;

    std::string type = argv[1];
    std::size_t n = std::stoull(argv[2]);
    if (type == "int") run<int>(n);
    else if (type == "double") run<double>(n);
    else if (type == "char") run<char>(n);
    else if (type == "point")  run<Point>(n);
    else return 1;

    return 0;
}