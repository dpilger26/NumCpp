/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2025 David Pilger
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
/// Mirror boundary
///
#pragma once

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Slice.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/flipud.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc::filter::boundary
{
    //============================================================================
    // Method Description:
    /// Mirror boundary
    ///
    /// @param inImage
    /// @param inBoundarySize
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> mirror2d(const NdArray<dtype>& inImage, uint32 inBoundarySize)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const Shape inShape = inImage.shape();
        Shape       outShape(inShape);
        outShape.rows += inBoundarySize * 2;
        outShape.cols += inBoundarySize * 2;

        NdArray<dtype> outArray(outShape);
        outArray.put(Slice(inBoundarySize, inBoundarySize + inShape.rows),
                     Slice(inBoundarySize, inBoundarySize + inShape.cols),
                     inImage);

        for (uint32 row = 0; row < inBoundarySize; ++row)
        {
            // bottom
            outArray.put(row,
                         Slice(inBoundarySize, inBoundarySize + inShape.cols),
                         inImage(inBoundarySize - row, Slice(0, inShape.cols)));

            // top
            outArray.put(row + inBoundarySize + inShape.rows,
                         Slice(inBoundarySize, inBoundarySize + inShape.cols),
                         inImage(inShape.rows - row - 2, Slice(0, inShape.cols)));
        }

        for (uint32 col = 0; col < inBoundarySize; ++col)
        {
            // left
            outArray.put(Slice(inBoundarySize, inBoundarySize + inShape.rows),
                         col,
                         inImage(Slice(0, inShape.rows), inBoundarySize - col));

            // right
            outArray.put(Slice(inBoundarySize, inBoundarySize + inShape.rows),
                         col + inBoundarySize + inShape.cols,
                         inImage(Slice(0, inShape.rows), inShape.cols - col - 2));
        }

        // now fill in the corners
        NdArray<dtype> lowerLeft =
            flipud(outArray(Slice(inBoundarySize + 1, 2 * inBoundarySize + 1), Slice(0, inBoundarySize)));
        NdArray<dtype> lowerRight = flipud(outArray(Slice(inBoundarySize + 1, 2 * inBoundarySize + 1),
                                                    Slice(outShape.cols - inBoundarySize, outShape.cols)));

        const uint32   upperRowStart = outShape.rows - 2 * inBoundarySize - 1;
        NdArray<dtype> upperLeft =
            flipud(outArray(Slice(upperRowStart, upperRowStart + inBoundarySize), Slice(0, inBoundarySize)));
        NdArray<dtype> upperRight = flipud(outArray(Slice(upperRowStart, upperRowStart + inBoundarySize),
                                                    Slice(outShape.cols - inBoundarySize, outShape.cols)));

        outArray.put(Slice(0, inBoundarySize), Slice(0, inBoundarySize), lowerLeft);
        outArray.put(Slice(0, inBoundarySize), Slice(outShape.cols - inBoundarySize, outShape.cols), lowerRight);
        outArray.put(Slice(outShape.rows - inBoundarySize, outShape.rows), Slice(0, inBoundarySize), upperLeft);
        outArray.put(Slice(outShape.rows - inBoundarySize, outShape.rows),
                     Slice(outShape.cols - inBoundarySize, outShape.cols),
                     upperRight);

        return outArray;
    }
} // namespace nc::filter::boundary
