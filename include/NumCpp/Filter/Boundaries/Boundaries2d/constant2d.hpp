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
/// Constant boundary
///
#pragma once

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Slice.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Filter/Boundaries/Boundaries2d/fillCorners.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc::filter::boundary
{
    //============================================================================
    // Method Description:
    /// Constant boundary
    ///
    /// @param inImage
    /// @param inBoundarySize
    /// @param inConstantValue
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> constant2d(const NdArray<dtype>& inImage, uint32 inBoundarySize, dtype inConstantValue)
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
        fillCorners(outArray, inBoundarySize, inConstantValue);

        outArray.put(Slice(0, inBoundarySize),
                     Slice(inBoundarySize, inBoundarySize + inShape.cols),
                     inConstantValue); /// bottom
        outArray.put(Slice(outShape.rows - inBoundarySize, outShape.rows),
                     Slice(inBoundarySize, inBoundarySize + inShape.cols),
                     inConstantValue); /// top
        outArray.put(Slice(inBoundarySize, inBoundarySize + inShape.rows),
                     Slice(0, inBoundarySize),
                     inConstantValue); /// left
        outArray.put(Slice(inBoundarySize, inBoundarySize + inShape.rows),
                     Slice(outShape.cols - inBoundarySize, outShape.cols),
                     inConstantValue); /// right

        return outArray;
    }
} // namespace nc::filter::boundary
