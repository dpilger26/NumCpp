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
/// Window expand around exceedance pixels
///

#pragma once

#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

#include <cmath>

namespace nc
{
    namespace imageProcessing
    {
        //============================================================================
        // Method Description:
        ///						Window expand around exceedance pixels
        ///
        /// @param				inExceedances
        /// @param				inBorderWidth
        /// @return
        ///				NdArray<bool>
        ///
        inline NdArray<bool> windowExceedances(const NdArray<bool>& inExceedances, uint8 inBorderWidth) 
        {
            // not the most efficient way to do things, but the easist...
            NdArray<bool> xcds(inExceedances);
            const Shape inShape = xcds.shape();
            for (uint8 border = 0; border < inBorderWidth; ++border)
            {
                for (int32 row = 0; row < static_cast<int32>(inShape.rows); ++row)
                {
                    for (int32 col = 0; col < static_cast<int32>(inShape.cols); ++col)
                    {
                        if (inExceedances(row, col))
                        {
                            xcds(std::max(row - 1, 0), std::max(col - 1, 0)) = true;
                            xcds(std::max(row - 1, 0), col) = true;
                            xcds(std::max(row - 1, 0), std::min<int32>(col + 1, inShape.cols - 1)) = true;

                            xcds(row, std::max<int32>(col - 1, 0)) = true;
                            xcds(row, std::min<int32>(col + 1, inShape.cols - 1)) = true;

                            xcds(std::min<int32>(row + 1, inShape.rows - 1), std::max(col - 1, 0)) = true;
                            xcds(std::min<int32>(row + 1, inShape.rows - 1), col) = true;
                            xcds(std::min<int32>(row + 1, inShape.rows - 1), std::min<int32>(col + 1, inShape.cols - 1)) = true;
                        }
                    }
                }
            }

            return xcds;
        }
    }
}
