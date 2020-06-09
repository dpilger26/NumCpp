/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 2.0.0
///
/// @section License
/// Copyright 2020 David Pilger
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
/// @section Description
/// Functions for working with NdArrays
///
#pragma once

#include "NumCpp/NdArray.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/mean.hpp"
#include "NumCpp/Utils/sqr.hpp"

#include <algorithm>
#include <cmath>
#include <complex>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Compute the standard deviation along the specified axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.std.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtype, class Alloc>
    NdArray<double, Alloc> stdev(const NdArray<dtype, Alloc>& inArray, Axis inAxis = Axis::NONE) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        double meanValue = 0.0;
        double sum = 0.0;

        const auto function = [&sum, &meanValue](dtype value) noexcept-> void
        {
            sum += utils::sqr(static_cast<double>(value) - meanValue);
        };

        switch (inAxis)
        {
            case Axis::NONE:
            {
                meanValue = mean(inArray, inAxis).item();
                std::for_each(inArray.cbegin(), inArray.cend(), function);

                NdArray<double, Alloc> returnArray = { std::sqrt(sum / inArray.size()) };
                return returnArray;
            }
            case Axis::COL:
            {
                NdArray<double, Alloc> meanValueArray = mean(inArray, inAxis);
                NdArray<double, Alloc> returnArray(1, inArray.numRows());
                for (uint32 row = 0; row < inArray.numRows(); ++row)
                {
                    meanValue = meanValueArray[row];
                    sum = 0.0;
                    std::for_each(inArray.cbegin(row), inArray.cend(row), function);

                    returnArray(0, row) = std::sqrt(sum / inArray.numCols());
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                NdArray<double, Alloc> meanValueArray = mean(inArray, inAxis);
                NdArray<dtype, Alloc> transposedArray = inArray.transpose();
                NdArray<double, Alloc> returnArray(1, transposedArray.numRows());
                for (uint32 row = 0; row < transposedArray.numRows(); ++row)
                {
                    meanValue = meanValueArray[row];
                    sum = 0.0;
                    std::for_each(transposedArray.cbegin(row), transposedArray.cend(row), function);

                    returnArray(0, row) = std::sqrt(sum / transposedArray.numCols());
                }

                return returnArray;
            }
            default:
            {
                return NdArray<double, Alloc>(); // get rid of compiler warning
            }
        }
    }

    //============================================================================
    // Method Description:
    ///						Compute the standard deviation along the specified axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.std.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtype, class Alloc>
    NdArray<std::complex<double>, Alloc> stdev(const NdArray<std::complex<dtype>, Alloc>& inArray,
        Axis inAxis = Axis::NONE) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        std::complex<double> meanValue(0.0, 0.0);
        std::complex<double> sum(0.0, 0.0);

        const auto function = [&sum, &meanValue](std::complex<dtype> value) noexcept-> void
        {
            sum += utils::sqr(complex_cast<double>(value) - meanValue);
        };

        switch (inAxis)
        {
            case Axis::NONE:
            {
                meanValue = mean(inArray, inAxis).item();
                std::for_each(inArray.cbegin(), inArray.cend(), function);

                NdArray<std::complex<double>, Alloc> returnArray = { std::sqrt(sum / static_cast<double>(inArray.size())) };
                return returnArray;
            }
            case Axis::COL:
            {
                NdArray<std::complex<double>, Alloc> meanValueArray = mean(inArray, inAxis);
                NdArray<std::complex<double>, Alloc> returnArray(1, inArray.numRows());
                for (uint32 row = 0; row < inArray.numRows(); ++row)
                {
                    meanValue = meanValueArray[row];
                    sum = std::complex<double>(0.0, 0.0);
                    std::for_each(inArray.cbegin(row), inArray.cend(row), function);

                    returnArray(0, row) = std::sqrt(sum / static_cast<double>(inArray.numCols()));
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                NdArray<std::complex<double>, Alloc> meanValueArray = mean(inArray, inAxis);
                NdArray<std::complex<dtype>, Alloc> transposedArray = inArray.transpose();
                NdArray<std::complex<double>, Alloc> returnArray(1, transposedArray.numRows());
                for (uint32 row = 0; row < transposedArray.numRows(); ++row)
                {
                    meanValue = meanValueArray[row];
                    sum = std::complex<double>(0.0, 0.0);
                    std::for_each(transposedArray.cbegin(row), transposedArray.cend(row), function);

                    returnArray(0, row) = std::sqrt(sum / static_cast<double>(transposedArray.numCols()));
                }

                return returnArray;
            }
            default:
            {
                return NdArray<std::complex<double>>(); // get rid of compiler warning
            }
        }
    }
}
