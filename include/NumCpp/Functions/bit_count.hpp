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

#include "NumCpp/Core/DtypeInfo.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Computes the number of 1-bits in an integer
    ///
    /// @param inValue
    /// @return value
    ///
    template<typename dtype>
    constexpr int bit_count(dtype inValue) noexcept
    {
        STATIC_ASSERT_UNSIGNED_INTEGER(dtype);

        int count = 0;
        for (int bit = 0; bit < DtypeInfo<dtype>::bits(); ++bit)
        {
            count += static_cast<int>((inValue & (dtype{ 1 } << bit)) >> bit);
        }

        return count;
    }

    //============================================================================
    // Method Description:
    /// Computes the number of 1-bits in an integer
    ///
    /// @param inArray
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<int> bit_count(const NdArray<dtype>& inArray)
    {
        NdArray<int> returnArray(inArray.shape());

        stl_algorithms::transform(inArray.cbegin(),
                                  inArray.cend(),
                                  returnArray.begin(),
                                  [](dtype inValue) noexcept -> int { return bit_count(inValue); });

        return returnArray;
    }
} // namespace nc
