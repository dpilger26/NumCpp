#include"NumCpp.hpp"

#include<iostream>

int main()
{
    NC::NdArray<int> a = { {1, 2}, {3, 4} };
    a.print();
    std::cout << std::endl;

    NC::NdArray<int> b = { {1, 2}, {3, 4}, {5, 6} };
    b.print();
    std::cout << std::endl;

    b.reshape(2, 3);
    b.print();
    std::cout << std::endl;

    return 0;
}