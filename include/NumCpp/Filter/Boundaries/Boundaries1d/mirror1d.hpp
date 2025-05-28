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
/// Mirror boundary1d
///
#pragma once

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Slice.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/fliplr.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc::filter::boundary
{
    //============================================================================
    // Method Description:
    /// Mirror boundary1d
    ///
    /// @param inImage
    /// @param inBoundarySize
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> mirror1d(const NdArray<dtype>& inImage, uint32 inBoundarySize)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const uint32 outSize = inImage.size() + inBoundarySize * 2;

        NdArray<dtype> outArray(1, outSize);
        outArray.put(Slice(inBoundarySize, inBoundarySize + inImage.size()), inImage);

        // left
        outArray.put(Slice(0, inBoundarySize), fliplr(inImage[Slice(1, inBoundarySize + 1)]));

        // right
        outArray.put(Slice(inImage.size() + inBoundarySize, outSize),
                     fliplr(inImage[Slice(-static_cast<int32>(inBoundarySize) - 1, -1)]));

        return outArray;
    }
} // namespace nc::filter::boundary
