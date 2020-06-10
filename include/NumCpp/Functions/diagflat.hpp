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
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"

#include <cmath>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Create a two-dimensional array with the flattened input as a diagonal.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.diagflat.html
    ///
    /// @param      inArray
    /// @param      k Diagonal to set; 0, the default, corresponds to the “main” diagonal, 
    ///             a positive (negative) k giving the number of the diagonal above (below) the main.
    ///
    /// @return     NdArray
    ///
    template<typename dtype>
    NdArray<dtype> diagflat(const NdArray<dtype>& inArray, int32 k = 0) noexcept
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const uint32 absK = static_cast<uint32>(std::abs(k));
        NdArray<dtype> returnArray(inArray.size() + absK);

        const uint32 rowOffset = k < 0 ? absK : 0;
        const uint32 colOffset = k > 0 ? absK : 0;

        returnArray.zeros();
        for (uint32 i = 0; i < inArray.size(); ++i)
        {
            returnArray(i + rowOffset, i + colOffset) = inArray[i];
        }

        return returnArray;
    }
}
