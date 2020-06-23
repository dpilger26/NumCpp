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

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/NdArray.hpp"


#include <string>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Return evenly spaced numbers over a specified interval.
    ///
    ///						Returns num evenly spaced samples, calculated over the
    ///						interval[start, stop].
    ///
    ///						The endpoint of the interval can optionally be excluded.
    ///
    ///						Mostly only usefull if called with a floating point type
    ///						for the template argument.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.linspace.html
    ///
    /// @param				inStart
    /// @param				inStop
    /// @param				inNum: number of points (default = 50)
    /// @param				endPoint: include endPoint (default = true)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> linspace(dtype inStart, dtype inStop, uint32 inNum = 50, bool endPoint = true)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (inNum == 0)
        {
            return NdArray<dtype>(0);
        }

        if (inNum == 1)
        {
            NdArray<dtype> returnArray = { inStart };
            return returnArray;
        }

        if (inStop <= inStart)
        {
            THROW_INVALID_ARGUMENT_ERROR("stop value must be greater than the start value.");
        }

        if (endPoint)
        {
            if (inNum == 2)
            {
                NdArray<dtype> returnArray = { inStart, inStop };
                return returnArray;
            }
            
            NdArray<dtype> returnArray(1, inNum);
            returnArray.front() = inStart;
            returnArray.back() = inStop;

            dtype step = (inStop - inStart) / static_cast<dtype>(inNum - 1);

            for (uint32 i = 1; i < inNum - 1; ++i)
            {
                returnArray[i] = inStart + static_cast<dtype>(i) * step;
            }

            return returnArray;
        }
        
        if (inNum == 2)
        {
            dtype step = (inStop - inStart) / (inNum);
            NdArray<dtype> returnArray = { inStart, inStart + step };
            return returnArray;
        }
        
        NdArray<dtype> returnArray(1, inNum);
        returnArray.front() = inStart;

        dtype step = (inStop - inStart) / static_cast<dtype>(inNum);

        for (uint32 i = 1; i < inNum; ++i)
        {
            returnArray[i] = inStart + static_cast<dtype>(i) * step;
        }

        return returnArray;
    }
}  // namespace nc
