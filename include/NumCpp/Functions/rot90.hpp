/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2026 David Pilger
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

#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/flip.hpp"
#include "NumCpp/Functions/fliplr.hpp"
#include "NumCpp/Functions/flipud.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Rotate an array by 90 degrees counter clockwise in the plane.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.rot90.html
    ///
    /// @param inArray
    /// @param inK: the number of times to rotate 90 degrees
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> rot90(const NdArray<dtype>& inArray, uint8 inK = 1)
    {
        inK %= 4;
        switch (inK)
        {
            case 0:
            {
                return inArray;
            }
            case 1:
            {
                return flipud(inArray.transpose());
            }
            case 2:
            {
                return flip(inArray, Axis::NONE);
            }
            case 3:
            {
                return fliplr(inArray.transpose());
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return {};
            }
        }
    }
} // namespace nc
