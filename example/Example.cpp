#include"NumCpp.hpp"

#include<iostream>

int main()
{
    // Containers
    NC::NdArray<int> a0 = { {1, 2}, {3, 4} };
    a0.print();
    std::cout << std::endl;

    NC::NdArray<int> a1 = { {1, 2}, {3, 4}, {5, 6} };
    a1.print();
    std::cout << std::endl;

    a1.reshape(2, 3);
    a1.print();
    std::cout << std::endl;

    auto a2 = a1.astype<double>();
    a2.print();
    std::cout << std::endl;

    // Initializers
    auto a3 = NC::linspace<int>(1, 10, 5);
    a3.print();
    std::cout << std::endl;

    auto a4 = NC::arange<int>(3, 7);
    a4.print();
    std::cout << std::endl;

    auto a5 = NC::eye<int>(4);
    a5.print();
    std::cout << std::endl;

    auto a6 = NC::zeros<int>(3, 4);
    a6.print();
    std::cout << std::endl;

    auto a7 = NC::NdArray<int>(3, 4) = 0;
    a7.print();
    std::cout << std::endl;

    auto a8 = NC::ones<int>(3, 4);
    a8.print();
    std::cout << std::endl;

    auto a9 = NC::NdArray<int>(3, 4) = 1;
    a9.print();
    std::cout << std::endl;

    auto a10 = NC::nans<double>(3, 4);
    a10.print();
    std::cout << std::endl;

    auto a11 = NC::NdArray<double>(3, 4) = NC::Constants::nan;
    a11.print();
    std::cout << std::endl;

    auto a12 = NC::empty(3, 4);
    a12.print();
    std::cout << std::endl;

    auto a13 = NC::NdArray<int>(3, 4);
    a12.print();
    std::cout << std::endl;

    // Slicing/Broadcasting
    auto a14 = NC::Random<int>::randInt(NC::Shape(10, 10), 0, 100);
    a14.print();
    std::cout << std::endl;

    std::cout << a14(2, 3) << std::endl;
    std::cout << a14(NC::Slice(2, 5), NC::Slice(5, 8)) << std::endl;
    std::cout << a14(a14.rSlice(), 7) << std::endl;
    std::cout << a14[a14 > 50] << std::endl;
    std::cout << a14.putMask(a14 > 50, 666) << std::endl;

    // Random

    return 0;
}