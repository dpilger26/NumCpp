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

#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Evenly round to the given number of decimals.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.around.html
    ///
    /// @param inValue
    /// @param inNumDecimals: (Optional, default = 0)
    /// @return value
    ///
    template<typename dtype>
    dtype around(dtype inValue, uint8 inNumDecimals = 0)
    {
        static_assert(nc::Arithmetic<dtype>, "Can only be used with arithmetic types");
        NdArray<dtype> value = { inValue };
        return value.round(inNumDecimals).item();
    }

    //============================================================================
    // Method Description:
    /// Evenly round to the given number of decimals.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.around.html
    ///
    /// @param inArray
    /// @param inNumDecimals: (Optional, default = 0)
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> around(const NdArray<dtype>& inArray, uint8 inNumDecimals = 0)
    {
        static_assert(nc::Arithmetic<dtype>, "Can only be used with arithmetic types");
        return inArray.round(inNumDecimals);
    }
} // namespace nc
