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

#include <cmath>
#include <string>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Trigonometric inverse tangent.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.arctan2.html
    ///
    /// @param inY
    /// @param inX
    /// @return value
    ///
    template<typename dtype>
    auto arctan2(dtype inY, dtype inX) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return std::atan2(inY, inX);
    }

    //============================================================================
    // Method Description:
    /// Trigonometric inverse tangent, element-wise.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.arctan2.html
    ///
    /// @param inY
    /// @param inX
    /// @return NdArray
    ///
    template<typename dtype>
    auto arctan2(const NdArray<dtype>& inY, const NdArray<dtype>& inX)
    {
        if (inX.shape() != inY.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("input array shapes are not consistant.");
        }

        NdArray<decltype(arctan2(dtype{ 0 }, dtype{ 0 }))> returnArray(inY.shape());
        stl_algorithms::transform(inY.cbegin(),
                                  inY.cend(),
                                  inX.cbegin(),
                                  returnArray.begin(),
                                  [](dtype y, dtype x) noexcept -> auto { return arctan2(y, x); });

        return returnArray;
    }
} // namespace nc
