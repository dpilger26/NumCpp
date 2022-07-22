/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2022 David Pilger
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
/// Functions for working with NdArrays
///
#pragma once

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Return a 2-D array with ones on the diagonal and zeros elsewhere.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.eye.html
    ///
    /// @param inN: number of rows (N)
    /// @param inM: number of columns (M)
    /// @param inK: Index of the diagonal: 0 (the default) refers to the main diagonal,
    /// a positive value refers to an upper diagonal, and a negative value to a lower diagonal.
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> eye(uint32 inN, uint32 inM, int32 inK = 0)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        NdArray<dtype> returnArray(inN, inM);
        returnArray.zeros();

        if (inK < 0)
        {
            uint32 col = 0;
            for (uint32 row = inK; row < inN; ++row)
            {
                if (col >= inM)
                {
                    break;
                }

                returnArray(row, col++) = dtype{ 1 };
            }
        }
        else
        {
            uint32 row = 0;
            for (uint32 col = inK; col < inM; ++col)
            {
                if (row >= inN)
                {
                    break;
                }

                returnArray(row++, col) = dtype{ 1 };
            }
        }

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Return a 2-D array with ones on the diagonal and zeros elsewhere.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.eye.html
    ///
    /// @param inN: number of rows and columns (N)
    /// @param inK: Index of the diagonal: 0 (the default) refers to the main diagonal,
    /// a positive value refers to an upper diagonal, and a negative value to a lower diagonal.
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> eye(uint32 inN, int32 inK = 0)
    {
        return eye<dtype>(inN, inN, inK);
    }

    //============================================================================
    // Method Description:
    /// Return a 2-D array with ones on the diagonal and zeros elsewhere.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.eye.html
    ///
    /// @param inShape
    /// @param inK: Index of the diagonal: 0 (the default) refers to the main diagonal,
    /// a positive value refers to an upper diagonal, and a negative value to a lower diagonal.
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> eye(const Shape& inShape, int32 inK = 0)
    {
        return eye<dtype>(inShape.rows, inShape.cols, inK);
    }
} // namespace nc
