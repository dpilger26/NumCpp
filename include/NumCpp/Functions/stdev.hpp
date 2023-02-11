/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2023 David Pilger
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

#include <algorithm>
#include <cmath>
#include <complex>

#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/mean.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Utils/sqr.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Compute the standard deviation along the specified axis.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.std.html
    ///
    /// @param inArray
    /// @param inAxis (Optional, default NONE)
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<double> stdev(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        double meanValue = 0.;
        double sum       = 0.;

        const auto function = [&sum, &meanValue](dtype value) -> void
        { sum += utils::sqr(static_cast<double>(value) - meanValue); };

        switch (inAxis)
        {
            case Axis::NONE:
            {
                meanValue = mean(inArray, inAxis).item();
                std::for_each(inArray.cbegin(), inArray.cend(), function);

                NdArray<double> returnArray = { std::sqrt(sum / inArray.size()) };
                return returnArray;
            }
            case Axis::COL:
            {
                NdArray<double> meanValueArray = mean(inArray, inAxis);
                NdArray<double> returnArray(1, inArray.numRows());
                for (uint32 row = 0; row < inArray.numRows(); ++row)
                {
                    meanValue = meanValueArray[row];
                    sum       = 0.;
                    std::for_each(inArray.cbegin(row), inArray.cend(row), function);

                    returnArray(0, row) = std::sqrt(sum / inArray.numCols());
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return stdev(inArray.transpose(), Axis::COL);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
        }
    }

    //============================================================================
    // Method Description:
    /// Compute the standard deviation along the specified axis.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.std.html
    ///
    /// @param inArray
    /// @param inAxis (Optional, default NONE)
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<double>> stdev(const NdArray<std::complex<dtype>>& inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        std::complex<double> meanValue(0., 0.);
        std::complex<double> sum(0., 0.);

        const auto function = [&sum, &meanValue](std::complex<dtype> value) -> void
        { sum += utils::sqr(complex_cast<double>(value) - meanValue); };

        switch (inAxis)
        {
            case Axis::NONE:
            {
                meanValue = mean(inArray, inAxis).item();
                std::for_each(inArray.cbegin(), inArray.cend(), function);

                NdArray<std::complex<double>> returnArray = { std::sqrt(sum / static_cast<double>(inArray.size())) };
                return returnArray;
            }
            case Axis::COL:
            {
                NdArray<std::complex<double>> meanValueArray = mean(inArray, inAxis);
                NdArray<std::complex<double>> returnArray(1, inArray.numRows());
                for (uint32 row = 0; row < inArray.numRows(); ++row)
                {
                    meanValue = meanValueArray[row];
                    sum       = std::complex<double>(0., 0.);
                    std::for_each(inArray.cbegin(row), inArray.cend(row), function);

                    returnArray(0, row) = std::sqrt(sum / static_cast<double>(inArray.numCols()));
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                NdArray<std::complex<double>> meanValueArray  = mean(inArray, inAxis);
                NdArray<std::complex<dtype>>  transposedArray = inArray.transpose();
                NdArray<std::complex<double>> returnArray(1, transposedArray.numRows());
                for (uint32 row = 0; row < transposedArray.numRows(); ++row)
                {
                    meanValue = meanValueArray[row];
                    sum       = std::complex<double>(0., 0.);
                    std::for_each(transposedArray.cbegin(row), transposedArray.cend(row), function);

                    returnArray(0, row) = std::sqrt(sum / static_cast<double>(transposedArray.numCols()));
                }

                return returnArray;
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
        }
    }
} // namespace nc
