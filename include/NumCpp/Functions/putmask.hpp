/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.4
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

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Changes elements of an array based on conditional and input values.
    ///
    ///						Sets a.flat[n] = values[n] for each n where mask.flat[n] == True.
    ///
    ///						If values is not the same size as a and mask then it will repeat.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.putmask.html
    ///
    /// @param				inArray
    /// @param				inMask
    /// @param				inValue
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& putmask(NdArray<dtype>& inArray, const NdArray<bool>& inMask, dtype inValue)
    {
        inArray.putMask(inMask, inValue);
        return inArray;
    }

    //============================================================================
    // Method Description:
    ///						Changes elements of an array based on conditional and input values.
    ///
    ///						Sets a.flat[n] = values[n] for each n where mask.flat[n] == True.
    ///
    ///						If values is not the same size as a and mask then it will repeat.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.putmask.html
    ///
    /// @param				inArray
    /// @param				inMask
    /// @param				inValues
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& putmask(NdArray<dtype>& inArray, const NdArray<bool>& inMask, const NdArray<dtype>& inValues)
    {
        inArray.putMask(inMask, inValues);
        return inArray;
    }
}
