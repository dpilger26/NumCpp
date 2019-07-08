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

#include "NumCpp/Core/Error.hpp"
#include "NumCpp/NdArray.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						minimum of inputs.
    ///
    ///						Compare two value and returns a value containing the
    ///						minima
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fmin.html
    ///
    /// @param				inValue1
    /// @param				inValue2
    /// @return
    ///				value
    ///
    template<typename dtype>
    dtype fmin(dtype inValue1, dtype inValue2) noexcept
    {
        return std::min(inValue1, inValue2);
    }

    //============================================================================
    // Method Description:
    ///						Element-wise minimum of array elements.
    ///
    ///						Compare two arrays and returns a new array containing the
    ///						element - wise minima
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fmin.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> fmin(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            std::string errStr = "ERROR: fmin: input array shapes are not consistant.";
            error::throwInvalidArgument(errStr);
        }

        NdArray<double> returnArray(inArray1.shape());

        std::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, dtype inValue2) noexcept -> double
            { return min(inValue1, inValue2); });

        return returnArray;
    }
}
