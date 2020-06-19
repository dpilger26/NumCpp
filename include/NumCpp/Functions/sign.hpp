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
#include "NumCpp/Core/Internal/StdComplexOperators.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"

#include <complex>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Returns an element-wise indication of the sign of a number.
    ///
    ///						The sign function returns - 1 if x < 0, 0 if x == 0, 1 if x > 0.
    ///						nan is returned for nan inputs.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.sign.html
    ///
    /// @param
    ///				inValue
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    int8 sign(dtype inValue) noexcept 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        if (inValue < dtype{ 0 })
        {
            return -1;
        }
        else if (inValue > dtype{ 0 })
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }

    //============================================================================
    // Method Description:
    ///						Returns an element-wise indication of the sign of a number.
    ///
    ///						The sign function returns - 1 if x < 0, 0 if x == 0, 1 if x > 0.
    ///						nan is returned for nan inputs.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.sign.html
    ///
    /// @param
    ///				inArray
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<int8> sign(const NdArray<dtype>& inArray) 
    {
        NdArray<int8> returnArray(inArray.shape());
        stl_algorithms::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> int8
            { 
                return sign(inValue); 
            });

        return returnArray;
    }
}
