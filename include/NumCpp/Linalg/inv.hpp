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
/// matrix inverse
///
#pragma once

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

#include <string>

namespace nc
{
    namespace linalg
    {
        //============================================================================
        // Method Description:
        ///						matrix inverse
        ///
        ///                     SciPy Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.inv.html#scipy.linalg.inv
        ///
        /// @param
        ///				inArray
        /// @return
        ///				NdArray
        ///
        template<typename dtype>
        NdArray<double> inv(const NdArray<dtype>& inArray)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            const Shape inShape = inArray.shape();
            if (inShape.rows != inShape.cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("input array must be square.");
            }

            const uint32 order = inShape.rows;

            Shape newShape(inShape);
            newShape.rows *= 2;
            newShape.cols *= 2;

            NdArray<double> tempArray(newShape);
            for (uint32 row = 0; row < order; ++row)
            {
                for (uint32 col = 0; col < order; ++col)
                {
                    tempArray(row, col) = static_cast<double>(inArray(row, col));
                }
            }

            for (uint32 row = 0; row < order; ++row)
            {
                for (uint32 col = order; col < 2 * order; ++col)
                {
                    if (row == col - order)
                    {
                        tempArray(row, col) = 1.0;
                    }
                    else
                    {
                        tempArray(row, col) = 0.0;
                    }
                }
            }

            for (uint32 row = 0; row < order; ++row)
            {
                double t = tempArray(row, row);
                for (uint32 col = row; col < 2 * order; ++col)
                {
                    tempArray(row, col) /= t;
                }

                for (uint32 col = 0; col < order; ++col)
                {
                    if (row != col)
                    {
                        t = tempArray(col, row);
                        for (uint32 k = 0; k < 2 * order; ++k)
                        {
                            tempArray(col, k) -= t * tempArray(row, k);
                        }
                    }
                }
            }

            NdArray<double> returnArray(inShape);
            for (uint32 row = 0; row < order; row++)
            {
                uint32 colCounter = 0;
                for (uint32 col = order; col < 2 * order; ++col)
                {
                    returnArray(row, colCounter++) = tempArray(row, col);
                }
            }

            return returnArray;
        }
    } // namespace linalg
}  // namespace nc
