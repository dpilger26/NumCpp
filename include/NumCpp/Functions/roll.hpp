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
/// Functions for working with NdArrays
///
#pragma once

#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

#include <cmath>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Roll array elements along a given axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.roll.html
    ///
    /// @param				inArray
    /// @param				inShift: (elements to shift, positive means forward, negative means backwards)
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype, class Alloc>
    NdArray<dtype, Alloc> roll(const NdArray<dtype, Alloc>& inArray, int32 inShift, Axis inAxis = Axis::NONE) noexcept
    {
        switch (inAxis)
        {
            case Axis::NONE:
            {
                uint32 shift = std::abs(inShift) % inArray.size();
                if (inShift > 0)
                {
                    shift = inArray.size() - shift;
                }

                NdArray<dtype, Alloc> returnArray(inArray);
                stl_algorithms::rotate(returnArray.begin(), returnArray.begin() + shift, returnArray.end());

                return returnArray;
            }
            case Axis::COL:
            {
                const Shape inShape = inArray.shape();

                uint32 shift = std::abs(inShift) % inShape.cols;
                if (inShift > 0)
                {
                    shift = inShape.cols - shift;
                }

                NdArray<dtype, Alloc> returnArray(inArray);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    stl_algorithms::rotate(returnArray.begin(row), returnArray.begin(row) + shift, returnArray.end(row));
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                const Shape inShape = inArray.shape();

                uint32 shift = std::abs(inShift) % inShape.rows;
                if (inShift > 0)
                {
                    shift = inShape.rows - shift;
                }

                NdArray<dtype, Alloc> returnArray = inArray.transpose();
                for (uint32 row = 0; row < inShape.cols; ++row)
                {
                    stl_algorithms::rotate(returnArray.begin(row), returnArray.begin(row) + shift, returnArray.end(row));
                }

                return returnArray.transpose();
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return NdArray<dtype, Alloc>(0);
            }
        }
    }
}
