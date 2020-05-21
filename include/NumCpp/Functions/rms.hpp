/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.4
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
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Utils/sqr.hpp"

#include <algorithm>
#include <cmath>
#include <complex>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Compute the root mean square (RMS) along the specified axis.
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> rms(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        double squareSum = 0.0;
        auto function = [&squareSum](dtype value) noexcept -> void
        {
            squareSum += utils::sqr(static_cast<double>(value));
        };

        switch (inAxis)
        {
            case Axis::NONE:
            {
                std::for_each(inArray.cbegin(), inArray.cend(), function);
                NdArray<double> returnArray = { std::sqrt(squareSum / static_cast<double>(inArray.size())) };
                return returnArray;
            }
            case Axis::COL:
            {
                NdArray<double> returnArray(1, inArray.numRows());
                for (uint32 row = 0; row < inArray.numRows(); ++row)
                {
                    squareSum = 0.0;
                    std::for_each(inArray.cbegin(row), inArray.cend(row), function);
                    returnArray(0, row) = std::sqrt(squareSum / static_cast<double>(inArray.numCols()));
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                NdArray<dtype> transposedArray = inArray.transpose();
                NdArray<double> returnArray(1, transposedArray.numRows());
                for (uint32 row = 0; row < transposedArray.numRows(); ++row)
                {
                    squareSum = 0.0;
                    std::for_each(transposedArray.cbegin(row), transposedArray.cend(row), function);
                    returnArray(0, row) = std::sqrt(squareSum / static_cast<double>(transposedArray.numCols()));
                }

                return returnArray;
            }
            default:
            {
                return NdArray<double>(); // get rid of compiler warning
            }
        }
    }

    //============================================================================
    // Method Description:
    ///						Compute the root mean square (RMS) along the specified axis.
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<double>> rms(const NdArray<std::complex<dtype>>& inArray, Axis inAxis = Axis::NONE) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        std::complex<double> squareSum = 0.0;
        auto function = [&squareSum](std::complex<dtype> value) noexcept -> void
        {
            squareSum += utils::sqr(complex_cast<double>(value));
        };

        switch (inAxis)
        {
            case Axis::NONE:
            {
                std::for_each(inArray.cbegin(), inArray.cend(), function);
                NdArray<std::complex<double>> returnArray = { std::sqrt(squareSum / static_cast<double>(inArray.size())) };
                return returnArray;
            }
            case Axis::COL:
            {
                NdArray<std::complex<double>> returnArray(1, inArray.numRows());
                for (uint32 row = 0; row < inArray.numRows(); ++row)
                {
                    squareSum = std::complex<double>(0.0, 0.0);
                    std::for_each(inArray.cbegin(row), inArray.cend(row), function);
                    returnArray(0, row) = std::sqrt(squareSum / static_cast<double>(inArray.numCols()));
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                NdArray<std::complex<dtype>> transposedArray = inArray.transpose();
                NdArray<std::complex<double>> returnArray(1, transposedArray.numRows());
                for (uint32 row = 0; row < transposedArray.numRows(); ++row)
                {
                    squareSum = std::complex<double>(0.0, 0.0);
                    std::for_each(transposedArray.cbegin(row), transposedArray.cend(row), function);
                    returnArray(0, row) = std::sqrt(squareSum / static_cast<double>(transposedArray.numCols()));
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
