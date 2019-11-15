/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.2
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

#include "NumCpp/NdArray.hpp"
#include "NumCpp/Core/Error.hpp"
#include "NumCpp/Core/StlAlgorithms.hpp"

#include <cmath>
#include <string>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						maximum of inputs.
    ///
    ///						Compare two value and returns a value containing the
    ///						maxima
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fmax.html
    ///
    /// @param				inValue1
    /// @param				inValue2
    /// @return
    ///				value
    ///
    template<typename dtype>
    dtype fmax(dtype inValue1, dtype inValue2) noexcept
    {
        return std::max(inValue1, inValue2);
    }

    //============================================================================
    // Method Description:
    ///						Element-wise maximum of array elements.
    ///
    ///						Compare two arrays and returns a new array containing the
    ///						element - wise maxima
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fmax.html
    ///
    /// @param				inArray1
    /// @param				inArray2
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> fmax(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        if (inArray1.shape() != inArray2.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("input array shapes are not consistant.");
        }

        NdArray<double> returnArray(inArray1.shape());

        stl_algorithms::transform(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), returnArray.begin(),
            [](dtype inValue1, dtype inValue2) noexcept -> double
            { 
                return fmax(inValue1, inValue2);
            });

        return returnArray;
    }
}
