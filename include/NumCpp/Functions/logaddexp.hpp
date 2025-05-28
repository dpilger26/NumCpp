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

#include <cmath>
#include <complex>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StdComplexOperators.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Logarithm of the sum of exponentiations of the inputs.
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html
    ///
    /// @param x1
    /// @param x2
    ///
    /// @return value
    ///
    template<typename dtype>
    auto logaddexp(dtype x1, dtype x2) noexcept
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return std::log(std::exp(x1) + std::exp(x2));
    }

    //============================================================================
    // Method Description:
    /// Logarithm of the sum of exponentiations of the inputs, element-wise.
    ///
    /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html
    ///
    /// @param x1
    /// @param x2
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    auto logaddexp(const NdArray<dtype>& x1, const NdArray<dtype>& x2)
    {
        if (x1.size() != x2.size())
        {
            THROW_INVALID_ARGUMENT_ERROR("Inputs 'x1', and 'x2' must be the same size");
        }

        NdArray<decltype(logaddexp(dtype{ 0 }, dtype{ 0 }))> returnArray(x1.shape());
        stl_algorithms::transform(x1.cbegin(),
                                  x1.cend(),
                                  x2.cbegin(),
                                  returnArray.begin(),
                                  [](dtype inX1, dtype inX2) noexcept -> auto { return logaddexp(inX1, inX2); });

        return returnArray;
    }
} // namespace nc
