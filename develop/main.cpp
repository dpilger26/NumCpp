#include "NdArrayCore.hpp"
#include "Utils.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace nc_develop;

int main()
{
    auto a = NdArray<int>(2, 3, 4);
    a.printInfo();

    a.reshape(4, 3, 2);
    a.printInfo();

    const std::vector<std::size_t> dims = {5, 6, 7};
    auto b = NdArray<int>(dims);
    b.printInfo();

    try
    {
        const std::vector<std::size_t> newDims = { 1, 2, 3 };
        b.reshape(newDims);
    }
    catch (const std::invalid_argument& err)
    {
        std::cout << err.what() << '\n';
    }

    std::cout << a(0) << '\n';
    std::cout << a(0, 0, 3) << '\n';
    std::cout << a(1, 1, 1) << '\n';

    try
    {
        std::cout << a(1, 1) << '\n';
    }
    catch (const std::invalid_argument& err)
    {
        std::cout << err.what() << '\n';
    }

    NdArray<int> array1d = { 1, 2, 3, 4, 5 };
    array1d.printInfo();
    utils::printContainer(array1d);

    NdArray<int> array2d = {
        {1, 2, 3},
        {4, 5, 6}, 
        {7, 8, 9} 
    };
    array2d.printInfo();
    utils::printContainer(array2d);

    NdArray<int> array3d = {
        { {1, 2, 3}, 
          {4, 5, 6}, 
          {7, 8, 9} },
        { {10, 11, 12}, 
          {13, 14, 15}, 
          {16, 17, 18} },
        { {19, 20, 21},
          {22, 23, 24},
          {25, 26, 27} } 
    };
    array3d.printInfo();
    utils::printContainer(array3d);

    return EXIT_SUCCESS;
}
