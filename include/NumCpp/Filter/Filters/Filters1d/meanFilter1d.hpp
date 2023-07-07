/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2023 David Pilger
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
/// Calculates a one-dimensional mean filter.
///
#pragma once

#include "NumCpp/Core/Slice.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Filter/Boundaries/Boundaries1d/addBoundary1d.hpp"
#include "NumCpp/Filter/Boundaries/Boundary.hpp"
#include "NumCpp/Functions/mean.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc::filter
{
    //============================================================================
    // Method Description:
    /// Calculates a one-dimensional mean filter.
    ///
    /// @param inImageArray
    /// @param inSize: linear size of the kernel to apply
    /// @param inBoundaryType: boundary mode (default Reflect) options (reflect, constant, nearest, mirror, wrap)
    /// @param inConstantValue: contant value if boundary = 'constant' (default 0)
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<double> meanFilter1d(const NdArray<dtype>& inImageArray,
                                 uint32                inSize,
                                 Boundary              inBoundaryType  = Boundary::REFLECT,
                                 dtype                 inConstantValue = 0)
    {
        NdArray<dtype> arrayWithBoundary =
            boundary::addBoundary1d(inImageArray, inBoundaryType, inSize, inConstantValue);
        NdArray<double> output(1, inImageArray.size());

        const uint32 boundarySize = inSize / 2; // integer division
        const uint32 endPoint     = boundarySize + inImageArray.size();

        for (uint32 i = boundarySize; i < endPoint; ++i)
        {
            NdArray<dtype> window = arrayWithBoundary[Slice(i - boundarySize, i + boundarySize + 1)];

            output[i - boundarySize] = mean(window).item();
        }

        return output;
    }
} // namespace nc::filter
