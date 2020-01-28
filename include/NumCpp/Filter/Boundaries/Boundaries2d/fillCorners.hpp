/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.3
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
/// extends the corner values
///
#pragma once

#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Slice.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    namespace filter
    {
        namespace boundary
        {
            //============================================================================
            // Method Description:
            ///						extends the corner values
            ///
            /// @param				inArray
            /// @param              inBorderWidth
            ///
            template<typename dtype>
            void fillCorners(NdArray<dtype>& inArray, uint32 inBorderWidth)
            {
                const Shape inShape = inArray.shape();
                const int32 numRows = static_cast<int32>(inShape.rows);
                const int32 numCols = static_cast<int32>(inShape.cols);

                // top left
                inArray.put(Slice(0, inBorderWidth), Slice(0, inBorderWidth),
                    inArray(inBorderWidth, inBorderWidth));

                // top right
                inArray.put(Slice(0, inBorderWidth), Slice(numCols - inBorderWidth, numCols),
                    inArray(inBorderWidth, numCols - inBorderWidth - 1));

                // bottom left
                inArray.put(Slice(numRows - inBorderWidth, numRows), Slice(0, inBorderWidth),
                    inArray(numRows - inBorderWidth - 1, inBorderWidth));

                // bottom right
                inArray.put(Slice(numRows - inBorderWidth, numRows), Slice(numCols - inBorderWidth, numCols),
                    inArray(numRows - inBorderWidth - 1, numCols - inBorderWidth - 1));
            }

            //============================================================================
            // Method Description:
            ///						extends the corner values
            ///
            /// @param				inArray
            /// @param              inBorderWidth
            /// @param				inFillValue
            ///
            template<typename dtype>
            void fillCorners(NdArray<dtype>& inArray, uint32 inBorderWidth, dtype inFillValue)
            {
                const Shape inShape = inArray.shape();
                const int32 numRows = static_cast<int32>(inShape.rows);
                const int32 numCols = static_cast<int32>(inShape.cols);

                // top left
                inArray.put(Slice(0, inBorderWidth), Slice(0, inBorderWidth), inFillValue);

                // top right
                inArray.put(Slice(0, inBorderWidth), Slice(numCols - inBorderWidth, numCols), inFillValue);

                // bottom left
                inArray.put(Slice(numRows - inBorderWidth, numRows), Slice(0, inBorderWidth), inFillValue);

                // bottom right
                inArray.put(Slice(numRows - inBorderWidth, numRows), Slice(numCols - inBorderWidth, numCols), inFillValue);
            }
        }
    }
}
