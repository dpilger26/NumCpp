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
    ///						Counts the number of non-zero values in the array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.count_nonzero.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<uint32> count_nonzero(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE) noexcept
    {
        switch (inAxis)
        {
            case Axis::NONE:
            {
                NdArray<uint32> count = { inArray.size() - 
                    static_cast<uint32>(stl_algorithms::count(inArray.cbegin(), inArray.cend(), dtype{ 0 })) };
                return count;
            }
            case Axis::COL:
            {
                Shape inShape = inArray.shape();

                NdArray<uint32> returnArray(1, inShape.rows);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    returnArray(0, row) = inShape.cols -
                        static_cast<uint32>(stl_algorithms::count(inArray.cbegin(row), inArray.cend(row), dtype{ 0 }));
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                NdArray<dtype> inArrayTranspose = inArray.transpose();
                Shape inShapeTransposed = inArrayTranspose.shape();
                NdArray<uint32> returnArray(1, inShapeTransposed.rows);
                for (uint32 row = 0; row < inShapeTransposed.rows; ++row)
                {
                    returnArray(0, row) = inShapeTransposed.cols -
                        static_cast<uint32>(stl_algorithms::count(inArrayTranspose.cbegin(row), inArrayTranspose.cend(row), dtype{ 0 }));
                }

                return returnArray;
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return NdArray<uint32>(0);
            }
        }
    }
}
