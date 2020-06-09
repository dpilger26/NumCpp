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
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"

#include <cmath>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Return the natural logarithm of one plus the input array.
    ///
    ///						Calculates log(1 + x).
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.log1p.html
    ///
    /// @param
    ///				inValue
    ///
    /// @return
    ///				value
    ///
    template<typename dtype>
    auto log1p(dtype inValue) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return std::log1p(inValue);
    }

    //============================================================================
    // Method Description:
    ///						Return the natural logarithm of one plus the input array, element-wise.
    ///
    ///						Calculates log(1 + x).
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.log1p.html
    ///
    /// @param
    ///				inArray
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype, class Alloc>
    auto log1p(const NdArray<dtype, Alloc>& inArray) noexcept
    {
        NdArray<decltype(log1p(dtype{0})), Alloc> returnArray(inArray.shape());
        stl_algorithms::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> auto
            { 
                return log1p(inValue); 
            });

        return returnArray;
    }
}
