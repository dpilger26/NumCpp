/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2023 David Pilger
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
#include "NumCpp/Functions/full.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Return a new array of given shape and type, filled with ones.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ones.html
    ///
    /// @param inSquareSize
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> ones(uint32 inSquareSize)
    {
        static_assert(nc::ArithmeticOrComplex<dtype>, "Can only be used with arithmetic or std::complex types");

        return full(inSquareSize, inSquareSize, dtype{ 1 });
    }

    //============================================================================
    // Method Description:
    /// Return a new array of given shape and type, filled with ones.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ones.html
    ///
    /// @param inNumRows
    /// @param inNumCols
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> ones(uint32 inNumRows, uint32 inNumCols)
    {
        static_assert(nc::ArithmeticOrComplex<dtype>, "Can only be used with arithmetic or std::complex types");

        return full(inNumRows, inNumCols, dtype{ 1 });
    }

    //============================================================================
    // Method Description:
    /// Return a new array of given shape and type, filled with ones.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.ones.html
    ///
    /// @param inShape
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> ones(const Shape& inShape)
    {
        static_assert(nc::ArithmeticOrComplex<dtype>, "Can only be used with arithmetic or std::complex types");

        return full(inShape, dtype{ 1 });
    }
} // namespace nc
