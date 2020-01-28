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

#include "NumCpp/Core/Error.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/StlAlgorithms.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

#include <string>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Return the gradient of the array.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.gradient.html
    ///
    ///
    /// @param				inArray
    /// @param				inAxis (default ROW)
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> gradient(const NdArray<dtype>& inArray, Axis inAxis = Axis::ROW)
    {
        switch (inAxis)
        {
            case Axis::ROW:
            {
                const auto inShape = inArray.shape();
                if (inShape.rows < 2)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input array must have more than 1 row.");
                }

                // first do the first and last rows
                auto returnArray = NdArray<double>(inShape);
                for (uint32 col = 0; col < inShape.cols; ++col)
                {
                    returnArray(0, col) = static_cast<double>(inArray(1, col)) - static_cast<double>(inArray(0, col));
                    returnArray(-1, col) = static_cast<double>(inArray(-1, col)) - static_cast<double>(inArray(-2, col));
                }

                // then rip through the rest of the array
                for (uint32 col = 0; col < inShape.cols; ++col)
                {
                    for (uint32 row = 1; row < inShape.rows - 1; ++row)
                    {
                        returnArray(row, col) = (static_cast<double>(inArray(row + 1, col)) - static_cast<double>(inArray(row - 1, col))) / 2.0;
                    }
                }

                return returnArray;
            }
            case Axis::COL:
            {
                const auto inShape = inArray.shape();
                if (inShape.cols < 2)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input array must have more than 1 columns.");
                }

                // first do the first and last columns
                auto returnArray = NdArray<double>(inShape);
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    returnArray(row, 0) = static_cast<double>(inArray(row, 1)) - static_cast<double>(inArray(row, 0));
                    returnArray(row, -1) = static_cast<double>(inArray(row, -1)) - static_cast<double>(inArray(row, -2));
                }

                // then rip through the rest of the array
                for (uint32 row = 0; row < inShape.rows; ++row)
                {
                    for (uint32 col = 1; col < inShape.cols - 1; ++col)
                    {
                        returnArray(row, col) = (static_cast<double>(inArray(row, col + 1)) - static_cast<double>(inArray(row, col - 1))) / 2.0;
                    }
                }

                return returnArray;
            }
            default:
            {
                // will return the gradient of the flattened array
                if (inArray.size() < 2)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input array must have more than 1 element.");
                }

                auto returnArray = NdArray<double>(1, inArray.size());
                returnArray[0] = static_cast<double>(inArray[1]) - static_cast<double>(inArray[0]);
                returnArray[-1] = static_cast<double>(inArray[-1]) - static_cast<double>(inArray[-2]);

                stl_algorithms::transform(inArray.cbegin() + 2, inArray.cend(), inArray.cbegin(), returnArray.begin() + 1,
                    [](dtype value1, dtype value2) noexcept -> double
                    { 
                        return (static_cast<double>(value1) - static_cast<double>(value2)) / 2.0; 
                    });

                return returnArray;
            }
        }
    }
}
