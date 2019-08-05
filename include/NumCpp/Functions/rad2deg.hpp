/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.1
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
/// Functions for working with NdArrays
///
#pragma once

#include "NumCpp/Core/Constants.hpp"
#include "NumCpp/NdArray.hpp"

#include <algorithm>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Convert angles from radians to degrees.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.rad2deg.html
    ///
    /// @param
    ///				inValue
    ///
    /// @return
    ///				value
    ///
    template<typename dtype>
    constexpr double rad2deg(dtype inValue) noexcept
    {
        return inValue * 180.0 / constants::pi;
    }

    //============================================================================
    // Method Description:
    ///						Convert angles from radians to degrees.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.rad2deg.html
    ///
    /// @param
    ///				inArray
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> rad2deg(const NdArray<dtype>& inArray) noexcept
    {
        NdArray<double> returnArray(inArray.shape());
        std::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> double
            { return rad2deg(inValue); });

        return returnArray;
    }
}
