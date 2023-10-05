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
/// Calculates a multidimensional percentile filter.
///
#pragma once

#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Slice.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Filter/Boundaries/Boundaries2d/addBoundary2d.hpp"
#include "NumCpp/Functions/percentile.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc::filter
{
    //============================================================================
    // Method Description:
    /// Calculates a multidimensional percentile filter.
    ///
    /// SciPy Reference:
    /// https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.percentile_filter.html#scipy.ndimage.percentile_filter
    ///
    /// @param inImageArray
    /// @param inSize: square size of the kernel to apply
    /// @param inPercentile: percentile [0, 100]
    /// @param inBoundaryType: boundary mode (default Reflect) options (reflect, constant, nearest, mirror, wrap)
    /// @param inConstantValue: contant value if boundary = 'constant' (default 0)
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> percentileFilter(const NdArray<dtype>& inImageArray,
                                    uint32                inSize,
                                    double                inPercentile,
                                    Boundary              inBoundaryType  = Boundary::REFLECT,
                                    dtype                 inConstantValue = 0)
    {
        NdArray<dtype> arrayWithBoundary =
            boundary::addBoundary2d(inImageArray, inBoundaryType, inSize, inConstantValue);
        NdArray<dtype> output(inImageArray.shape());

        const Shape  inShape      = inImageArray.shape();
        const uint32 boundarySize = inSize / 2; // integer division
        const uint32 endPointRow  = boundarySize + inShape.rows;
        const uint32 endPointCol  = boundarySize + inShape.cols;

        for (uint32 row = boundarySize; row < endPointRow; ++row)
        {
            for (uint32 col = boundarySize; col < endPointCol; ++col)
            {
                NdArray<dtype> window = arrayWithBoundary(Slice(row - boundarySize, row + boundarySize + 1),
                                                          Slice(col - boundarySize, col + boundarySize + 1));

                output(row - boundarySize, col - boundarySize) =
                    percentile(window, inPercentile, Axis::NONE, InterpolationMethod::NEAREST).item();
            }
        }

        return output;
    }
} // namespace nc::filter
