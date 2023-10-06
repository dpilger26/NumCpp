#include "NumCpp.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <vector>

using EigenIntMatrix    = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using EigenIntMatrixMap = Eigen::Map<EigenIntMatrix>;

int main()
{
    // construct some NumCpp arrays
    auto ncA = nc::random::randInt<int>({ 5, 5 }, 0, 10);
    auto ncB = nc::random::randInt<int>({ 5, 5 }, 0, 10);

    std::cout << "ncA:\n" << ncA << std::endl;
    std::cout << "ncB:\n" << ncB << std::endl;

    // map the arrays to Eigen
    auto eigenA = EigenIntMatrixMap(ncA.data(), ncA.numRows(), ncA.numCols());
    auto eigenB = EigenIntMatrixMap(ncB.data(), ncB.numRows(), ncB.numCols());

    // add the two Eigen matrices
    auto eigenC = eigenA + eigenB;

    // add the two NumCpp arrays for a sanity check
    auto ncC = ncA + ncB;

    // convert the Eigen result back to NumCpp
    auto data                                                    = std::vector<int>(eigenC.rows() * eigenC.cols());
    EigenIntMatrixMap(data.data(), eigenC.rows(), eigenC.cols()) = eigenC;

    auto ncCeigen = nc::NdArray<int>(data.data(), eigenC.rows(), eigenC.cols(), nc::PointerPolicy::SHELL);

    // compare the two outputs
    if (nc::array_equal(ncC, ncCeigen))
    {
        std::cout << "Arrays are equal." << std::endl;
        std::cout << ncC << std::endl;
    }
    else
    {
        std::cout << "Arrays are not equal." << std::endl;
        std::cout << "ncCeigen:\n" << ncCeigen << std::endl;
        std::cout << "ncC:\n" << ncC << std::endl;
    }

    return 0;
}
