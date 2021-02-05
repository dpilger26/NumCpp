#include "NumCpp.hpp"
#include "function.h"
#include <iostream>

int main()
{
    constexpr nc::uint32 numRows = 10;
    constexpr nc::uint32 numCols = 10;

    auto randArray1 = nc::random::rand<double>({numRows, numCols});
    std::cout << randArray1;

    auto randArray2 = getRandomArray(numRows, numCols);
    std::cout << randArray2;

    auto randArray3 = randArray1 + randArray2;
    std::cout << randArray3;

    return 0;
}
