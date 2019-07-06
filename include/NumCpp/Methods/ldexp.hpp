/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.0
///
/// @section License
/// Copyright 2019 David Pilger
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
/// Methods for working with NdArrays
///
#pragma once

#include"NumCpp/Core/Types.hpp"
#include"NumCpp/NdArray/NdArray.hpp"

#include<algorithm>
#include<cmath>
#include<iostream>
#include<string>
#include<stdexcept>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Returns x1 * 2^x2.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ldexp.html
    ///
    /// @param				inValue1
    /// @param				inValue2
    ///
    /// @return
    ///				value
    ///
    template<typename dtype>
    dtype ldexp(dtype inValue1, uint8 inValue2) noexcept
    {
        return static_cast<dtype>(std::ldexp(static_cast<double>(inValue1), inValue2));
    }

    //============================================================================
    // Method Description:
    ///						Returns x1 * 2^x2, element-wise.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ldexp.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> ldexp(const NdArray<dtype>& inArray1, const NdArray<uint8>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            std::string errStr = "ERROR: ldexp: input array shapes are not consistant.";
            std::cerr << errStr << std::endl;
            throw std::invalid_argument(errStr);
        }

        NdArray<dtype> returnArray(inArray1.shape());
        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, uint8 inValue2) noexcept -> dtype
        { return ldexp(inValue1, inValue2); });

        return returnArray;
    }
}
