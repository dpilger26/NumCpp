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
/// Methods for working with NdArrays
///
#pragma once

#include "NumCpp/Core/Error.hpp"
#include "NumCpp/NdArray.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Returns a boolean array where two arrays are element-wise
    ///						equal within a tolerance.
    ///
    ///						For finite values, isclose uses the following equation to test whether two floating point values are equivalent.
    ///						absolute(a - b) <= (atol + rtol * absolute(b))
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.isclose.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    /// @param				inRtol: relative tolerance (default 1e-5)
    /// @param				inAtol: absolute tolerance (default 1e-9)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<bool> isclose(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2, double inRtol = 1e-05, double inAtol = 1e-08)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            std::string errStr = "ERROR: isclose: input array shapes are not consistant.";
            error::throwInvalidArgument(errStr);
        }

        NdArray<bool> returnArray(inArray1.shape());
        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [inRtol, inAtol](dtype inValueA, dtype inValueB) noexcept -> bool
        { return std::abs(inValueA - inValueB) <= (inAtol + inRtol * std::abs(inValueB)); });

        return returnArray;
    }
}
