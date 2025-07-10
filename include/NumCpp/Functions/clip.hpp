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

#include <algorithm>

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StdComplexOperators.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Clip (limit) the value.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.clip.html
    ///
    /// @param inValue
    /// @param inMinValue
    /// @param inMaxValue
    /// @return NdArray
    ///
    template<typename dtype>
    dtype clip(dtype inValue, dtype inMinValue, dtype inMaxValue)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

#ifdef __cpp_lib_clamp
        const auto comparitor = [](dtype lhs, dtype rhs) noexcept -> bool { return lhs < rhs; };

        return std::clamp(inValue, inMinValue, inMaxValue, comparitor);
#else
        if (inValue < inMinValue)
        {
            return inMinValue;
        }
        else if (inValue > inMaxValue)
        {
            return inMaxValue;
        }

        return inValue;
#endif
    }

    //============================================================================
    // Method Description:
    /// Clip (limit) the values in an array.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.clip.html
    ///
    /// @param inArray
    /// @param inMinValue
    /// @param inMaxValue
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> clip(const NdArray<dtype>& inArray, dtype inMinValue, dtype inMaxValue)
    {
        return inArray.clip(inMinValue, inMaxValue);
    }
} // namespace nc
