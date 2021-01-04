/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2021 David Pilger
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
/// trims the boundary off to make the image back to the original size
///
#pragma once

#include "NumCpp/NdArray.hpp"
#include "NumCpp/Core/Slice.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"

namespace nc
{
    namespace filter
    {
        namespace boundary
        {
            //============================================================================
            // Method Description:
            ///						trims the boundary off to make the image back to the original size
            ///
            /// @param				inImageWithBoundary
            /// @param              inSize
            /// @return
            ///				NdArray
            ///
            template<typename dtype>
            NdArray<dtype> trimBoundary1d(const NdArray<dtype>& inImageWithBoundary, uint32 inSize)
            {
                STATIC_ASSERT_ARITHMETIC(dtype);

                uint32 boundarySize = inSize / 2; // integer division
                uint32 imageSize = inImageWithBoundary.size() - boundarySize * 2;

                return inImageWithBoundary[Slice(boundarySize, boundarySize + imageSize)];
            }
        }
    }
}
