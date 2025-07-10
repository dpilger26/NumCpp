/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2025 David Pilger
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy of this
/// software and associated documentation files(the "Software"), to deal in the Software
/// without restriction, including without limitation the rights to use, copy, modify,
/// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
/// permit persons to whom the Software is furnished to do so, subject to the following
/// conditions :
///
/// The above copyright notice and this permission notice shall be included in all copies
/// or substantial portions of the Software.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
/// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
/// PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
/// FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
/// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
/// DEALINGS IN THE SOFTWARE.
///
/// Description
/// Functions for working with NdArrays
///
#pragma once

#include <functional>
#include <numeric>
#include <vector>

#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Construct an array by executing a function over each coordinate.
    /// The resulting array therefore has a value fn(x) at coordinate(x).
    ///
    ///  NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.fromfunction.html
    ///
    ///  @param func: callable that accepts an integer coordinate and returns type T
    ///  @param size: the size of the 1d array to create
    ///
    ///  @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> fromfunction(const std::function<dtype(typename NdArray<dtype>::size_type)> func,
                                typename NdArray<dtype>::size_type                             size)
    {
        NdArray<dtype> result(1, size);
        const auto     indices = [size]
        {
            std::vector<typename NdArray<dtype>::size_type> temp(size);
            std::iota(temp.begin(), temp.end(), 0);
            return temp;
        }();

        stl_algorithms::transform(indices.begin(),
                                  indices.end(),
                                  result.begin(),
                                  [&func](const auto idx) { return func(idx); });

        return result;
    }

    //============================================================================
    // Method Description:
    /// Construct an array by executing a function over each coordinate.
    /// The resulting array therefore has a value fn(x, y) at coordinate(x, y).
    ///
    ///  NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.fromfunction.html
    ///
    ///  @param func: callable that accepts an integer coordinate and returns type T
    ///  @param shape: the shape of the array to create
    ///
    ///  @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> fromfunction(
        const std::function<dtype(typename NdArray<dtype>::size_type, typename NdArray<dtype>::size_type)> func,
        Shape                                                                                              shape)
    {
        NdArray<dtype> result(shape);
        const auto     rows = [&shape]
        {
            std::vector<typename NdArray<dtype>::size_type> temp(shape.rows);
            std::iota(temp.begin(), temp.end(), 0);
            return temp;
        }();
        const auto cols = [&shape]
        {
            std::vector<typename NdArray<dtype>::size_type> temp(shape.cols);
            std::iota(temp.begin(), temp.end(), 0);
            return temp;
        }();

        stl_algorithms::for_each(rows.begin(),
                                 rows.end(),
                                 [&cols, &result, &func](const auto row)
                                 {
                                     stl_algorithms::transform(cols.begin(),
                                                               cols.end(),
                                                               result.begin(row),
                                                               [&func, row](const auto col) { return func(row, col); });
                                 });

        return result;
    }
} // namespace nc
