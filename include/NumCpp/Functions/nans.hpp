/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2025 David Pilger
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

#include "NumCpp/Core/Constants.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Functions/full.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Return a new array of given shape and type, filled with nans.
    /// Only really works for dtype = float/double
    ///
    /// @param inSquareSize
    /// @return NdArray
    ///
    inline NdArray<double> nans(uint32 inSquareSize)
    {
        return full(inSquareSize, constants::nan);
    }

    //============================================================================
    // Method Description:
    /// Return a new array of given shape and type, filled with nans.
    /// Only really works for dtype = float/double
    ///
    /// @param inNumRows
    /// @param inNumCols
    /// @return NdArray
    ///
    inline NdArray<double> nans(uint32 inNumRows, uint32 inNumCols)
    {
        return full(inNumRows, inNumCols, constants::nan);
    }

    //============================================================================
    // Method Description:
    /// Return a new array of given shape and type, filled with nans.
    /// Only really works for dtype = float/double
    ///
    /// @param inShape
    /// @return NdArray
    ///
    inline NdArray<double> nans(const Shape& inShape)
    {
        return full(inShape, constants::nan);
    }
} // namespace nc
