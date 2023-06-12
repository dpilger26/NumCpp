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

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"
#include "NumCpp/Utils/sqr.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Matrix or vector norm.
    ///
    /// @param inArray
    /// @param inAxis (Optional, default NONE)
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<double> norm(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        double     sumOfSquares = 0.;
        const auto function     = [&sumOfSquares](dtype value) -> void
        { sumOfSquares += utils::sqr(static_cast<double>(value)); };

        switch (inAxis)
        {
            case Axis::NONE:
            {
                std::for_each(inArray.cbegin(), inArray.cend(), function);

                NdArray<double> returnArray = { std::sqrt(sumOfSquares) };
                return returnArray;
            }
            case Axis::COL:
            {
                NdArray<double> returnArray(1, inArray.numRows());
                for (uint32 row = 0; row < inArray.numRows(); ++row)
                {
                    sumOfSquares = 0.;
                    std::for_each(inArray.cbegin(row), inArray.cend(row), function);
                    returnArray(0, row) = std::sqrt(sumOfSquares);
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return norm(inArray.transpose(), Axis::COL);
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
    /// Matrix or vector norm.
    ///
    /// @param inArray
    /// @param inAxis (Optional, default NONE)
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<double>> norm(const NdArray<std::complex<dtype>>& inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        std::complex<double> sumOfSquares(0., 0.);
        const auto           function = [&sumOfSquares](const std::complex<dtype>& value) -> void
        { sumOfSquares += utils::sqr(complex_cast<double>(value)); };

        switch (inAxis)
        {
            case Axis::NONE:
            {
                std::for_each(inArray.cbegin(), inArray.cend(), function);

                NdArray<std::complex<double>> returnArray = { std::sqrt(sumOfSquares) };
                return returnArray;
            }
            case Axis::COL:
            {
                NdArray<std::complex<double>> returnArray(1, inArray.numRows());
                for (uint32 row = 0; row < inArray.numRows(); ++row)
                {
                    sumOfSquares = std::complex<double>(0., 0.);
                    std::for_each(inArray.cbegin(row), inArray.cend(row), function);
                    returnArray(0, row) = std::sqrt(sumOfSquares);
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return norm(inArray.transpose(), Axis::COL);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
        }
    }
} // namespace nc
