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

#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Gives a new shape to an array without changing its data.
    ///
    /// The new shape should be compatible with the original shape. If an single integer,
    /// then the result will be a 1-D array of that length. One shape dimension 
    /// can be -1. In this case, the value is inferred from the length of the 
    /// array and remaining dimensions. 
    ///
    /// @param				inArray
    /// @param				inSize
    ///
    /// @return
    /// NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& reshape(NdArray<dtype>& inArray, uint32 inSize)
    {
        inArray.reshape(inSize);
        return inArray;
    }

    //============================================================================
    // Method Description:
    /// Gives a new shape to an array without changing its data.
    ///
    /// The new shape should be compatible with the original shape. If an single integer,
    /// then the result will be a 1-D array of that length. One shape dimension 
    /// can be -1. In this case, the value is inferred from the length of the 
    /// array and remaining dimensions. 
    ///
    /// @param				inArray
    /// @param				inNumRows
    /// @param				inNumCols
    ///
    /// @return
    /// NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& reshape(NdArray<dtype>& inArray, int32 inNumRows, int32 inNumCols)
    {
        inArray.reshape(inNumRows, inNumCols);
        return inArray;
    }

    //============================================================================
    // Method Description:
    /// Gives a new shape to an array without changing its data.
    ///
    /// The new shape should be compatible with the original shape. If an single integer,
    /// then the result will be a 1-D array of that length. One shape dimension 
    /// can be -1. In this case, the value is inferred from the length of the 
    /// array and remaining dimensions. 
    ///
    /// @param				inArray
    /// @param				inNewShape
    ///
    /// @return
    /// NdArray
    ///
    template<typename dtype>
    NdArray<dtype>& reshape(NdArray<dtype>& inArray, const Shape& inNewShape)
    {
        inArray.reshape(inNewShape);
        return inArray;
    }
}  // namespace nc
