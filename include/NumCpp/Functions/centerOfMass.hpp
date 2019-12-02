/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.2
///
/// @section License
/// Copyright 2019 David Pilger
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
/// Functions for working with NdArrays
///
#pragma once

#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/StlAlgorithms.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Returns the center of mass of the array values along an axis.
    ///
    /// @param      inArray
    /// @param      inAxis (Optional, default NONE which is a 2d center of mass)
    /// @return     NdArray: if axis is NONE then a 1x2 array of the centroid row/col is returned.
    ///
    template<typename dtype>
    NdArray<double> centerOfMass(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE) noexcept
    {
        const Shape shape = inArray.shape();

        switch (inAxis)
        {
            case Axis::NONE:
            {
                double inten = 0.0;
                std::for_each(inArray.begin(), inArray.end(),
                    [&inten](dtype value) -> void
                    {
                        inten += static_cast<double>(value);
                    });

                // first get the row center
                double row = 0;
                for (uint32 rowIdx = 0; rowIdx < shape.rows; ++rowIdx)
                {
                    double rowSum = 0;
                    std::for_each(inArray.begin(rowIdx), inArray.end(rowIdx),
                        [&rowSum](dtype value) -> void
                        {
                            rowSum += static_cast<double>(value);
                        });

                    row += rowSum * static_cast<double>(rowIdx);
                }

                row /= inten;

                // then get the column center
                double col = 0;
                for (uint32 colIdx = 0; colIdx < shape.cols; ++colIdx)
                {
                    double colSum = 0;
                    for (uint32 rowIdx = 0; rowIdx < shape.rows; ++rowIdx)
                    {
                        colSum += static_cast<double>(inArray(rowIdx, colIdx));
                    }
                    col += colSum * static_cast<double>(colIdx);
                }

                col /= inten;

                return { row, col };
            }
            case Axis::ROW:
            {
                NdArray<double> returnArray(1, shape.cols);
                returnArray.zeros();

                const NdArray<double> inten = inArray.template astype<double>().sum(inAxis);

                for (uint32 colIdx = 0; colIdx < shape.cols; ++colIdx)
                {
                    for (uint32 rowIdx = 0; rowIdx < shape.rows; ++rowIdx)
                    {
                        returnArray(0, colIdx) += static_cast<double>(inArray(rowIdx, colIdx)) * static_cast<double>(rowIdx);
                    }

                    returnArray(0, colIdx) /= inten[colIdx];
                }

                return returnArray;
            }
            case Axis::COL:
            {
                NdArray<double> returnArray(1, shape.rows);
                returnArray.zeros();

                const NdArray<double> inten = inArray.template astype<double>().sum(inAxis);

                for (uint32 rowIdx = 0; rowIdx < shape.rows; ++rowIdx)
                {
                    for (uint32 colIdx = 0; colIdx < shape.cols; ++colIdx)
                    {
                        returnArray(0, rowIdx) += static_cast<double>(inArray(rowIdx, colIdx)) * static_cast<double>(colIdx);
                    }

                    returnArray(0, rowIdx) /= inten[rowIdx];
                }

                return returnArray;
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return NdArray<double>(0);
            }
        }
    }
}
