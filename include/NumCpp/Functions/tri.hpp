/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
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
    ///						An array with ones at and below the given diagonal and zeros elsewhere.
    ///
    /// @param				inN: number of rows and cols
    /// @param				inOffset: (the sub-diagonal at and below which the array is filled.
    ///						k = 0 is the main diagonal, while k < 0 is below it,
    ///						and k > 0 is above. The default is 0.)
    ///
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> tril(uint32 inN, int32 inOffset = 0) 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        uint32 rowStart = 0;
        uint32 colStart = 0;
        if (inOffset > 0)
        {
            colStart = inOffset;
        }
        else
        {
            rowStart = inOffset * -1;
        }

        NdArray<dtype> returnArray(inN);
        returnArray.zeros();
        for (uint32 row = rowStart; row < inN; ++row)
        {
            for (uint32 col = 0; col < row + colStart + 1 - rowStart; ++col)
            {
                if (col == inN)
                {
                    break;
                }

                returnArray(row, col) = dtype{ 1 };
            }
        }

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						An array with ones at and below the given diagonal and zeros elsewhere.
    ///
    /// @param				inN: number of rows
    /// @param				inM: number of columns
    /// @param				inOffset: (the sub-diagonal at and below which the array is filled.
    ///						k = 0 is the main diagonal, while k < 0 is below it,
    ///						and k > 0 is above. The default is 0.)
    ///
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> tril(uint32 inN, uint32 inM, int32 inOffset = 0) 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        uint32 rowStart = 0;
        uint32 colStart = 0;
        if (inOffset > 0)
        {
            colStart = inOffset;
        }
        else if (inOffset < 0)
        {
            rowStart = inOffset * -1;
        }

        NdArray<dtype> returnArray(inN, inM);
        returnArray.zeros();
        for (uint32 row = rowStart; row < inN; ++row)
        {
            for (uint32 col = 0; col < row + colStart + 1 - rowStart; ++col)
            {
                if (col == inM)
                {
                    break;
                }

                returnArray(row, col) = dtype{ 1 };
            }
        }

        return returnArray;
    }

    // forward declare
    template<typename dtype>
    NdArray<dtype> triu(uint32 inN, uint32 inM, int32 inOffset = 0) ;

    //============================================================================
    // Method Description:
    ///                     Lower triangle of an array.
    ///
    ///                     Return a copy of an array with elements above the k - th diagonal zeroed.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.tril.html
    ///
    /// @param				inArray: number of rows and cols
    /// @param				inOffset: (the sub-diagonal at and below which the array is filled.
    ///						k = 0 is the main diagonal, while k < 0 is below it,
    ///						and k > 0 is above. The default is 0.)
    ///
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> tril(const NdArray<dtype>& inArray, int32 inOffset = 0) 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const Shape inShape = inArray.shape();
        auto outArray = inArray.copy();
        outArray.putMask(triu<bool>(inShape.rows, inShape.cols, inOffset + 1), 0);
        return outArray;
    }

    //============================================================================
    // Method Description:
    ///						An array with ones at and above the given diagonal and zeros elsewhere.
    ///
    /// @param				inN: number of rows
    /// @param				inM: number of columns
    /// @param				inOffset: (the sub-diagonal at and above which the array is filled.
    ///						k = 0 is the main diagonal, while k < 0 is below it,
    ///						and k > 0 is above. The default is 0.)
    ///
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> triu(uint32 inN, uint32 inM, int32 inOffset) 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        // because i'm stealing the lines of code from tril and reversing it, this is necessary
        inOffset -= 1;

        uint32 rowStart = 0;
        uint32 colStart = 0;
        if (inOffset > 0)
        {
            colStart = inOffset;
        }
        else if (inOffset < 0)
        {
            rowStart = inOffset * -1;
        }

        NdArray<dtype> returnArray(inN, inM);
        returnArray.ones();
        for (uint32 row = rowStart; row < inN; ++row)
        {
            for (uint32 col = 0; col < row + colStart + 1 - rowStart; ++col)
            {
                if (col == inM)
                {
                    break;
                }

                returnArray(row, col) = dtype{ 0 };
            }
        }

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						An array with ones at and above the given diagonal and zeros elsewhere.
    ///
    /// @param				inN: number of rows and cols
    /// @param				inOffset: (the sub-diagonal at and above which the array is filled.
    ///						k = 0 is the main diagonal, while k < 0 is below it,
    ///						and k > 0 is above. The default is 0.)
    ///
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> triu(uint32 inN, int32 inOffset = 0) 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return tril<dtype>(inN, -inOffset).transpose();
    }

    //============================================================================
    // Method Description:
    ///                     Upper triangle of an array.
    ///
    ///                     Return a copy of an array with elements below the k - th diagonal zeroed.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.triu.html
    ///
    /// @param				inArray: number of rows and cols
    /// @param				inOffset: (the sub-diagonal at and below which the array is filled.
    ///						k = 0 is the main diagonal, while k < 0 is below it,
    ///						and k > 0 is above. The default is 0.)
    ///
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<dtype> triu(const NdArray<dtype>& inArray, int32 inOffset = 0) 
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const Shape inShape = inArray.shape();
        auto outArray = inArray.copy();
        outArray.putMask(tril<bool>(inShape.rows, inShape.cols, inOffset - 1), 0);
        return outArray;
    }
}  // namespace nc
