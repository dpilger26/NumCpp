/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2026 David Pilger
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
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Internal/TypeTraits.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Return the remainder of division.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fmod.html
    ///
    ///
    /// @param inValue1
    /// @param inValue2
    /// @return value
    ///
    template<typename dtype, std::enable_if_t<std::is_integral_v<dtype>, int> = 0>
    dtype fmod(dtype inValue1, dtype inValue2) noexcept
    {
        return inValue1 % inValue2;
    }

    //============================================================================
    // Method Description:
    /// Return the remainder of division.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fmod.html
    ///
    ///
    /// @param inValue1
    /// @param inValue2
    /// @return value
    ///
    template<typename dtype, std::enable_if_t<std::is_floating_point_v<dtype>, int> = 0>
    dtype fmod(dtype inValue1, dtype inValue2) noexcept
    {
        return std::fmod(inValue1, inValue2);
    }

    //============================================================================
    // Method Description:
    /// Return the element-wise remainder of division.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fmod.html
    ///
    ///
    /// @param inArray1
    /// @param inArray2
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> fmod(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return inArray1 % inArray2;
    }
} // namespace nc
