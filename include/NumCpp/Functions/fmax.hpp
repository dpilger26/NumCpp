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

#include <cmath>
#include <complex>
#include <string>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// maximum of inputs.
    ///
    /// Compare two value and returns a value containing the
    /// maxima
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fmax.html
    ///
    /// @param inValue1
    /// @param inValue2
    /// @return value
    ///
    template<typename dtype>
    dtype fmax(dtype inValue1, dtype inValue2) noexcept
    {
        static_assert(nc::ArithmeticOrComplex<dtype>, "Can only be used with arithmetic or std::complex types");

        return std::max(inValue1,
                        inValue2,
                        [](const dtype value1, const dtype value2) noexcept -> bool { return value1 < value2; });
    }

    //============================================================================
    // Method Description:
    /// Element-wise maximum of array elements.
    ///
    /// Compare two arrays and returns a new array containing the
    /// element - wise maxima
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fmax.html
    ///
    /// @param inArray1
    /// @param inArray2
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> fmax(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return broadcast::broadcaster<dtype>(inArray1,
                                             inArray2,
                                             [](dtype inValue1, dtype inValue2) noexcept -> dtype
                                             { return fmax(inValue1, inValue2); });
    }

    //============================================================================
    // Method Description:
    /// Element-wise maximum of array elements.
    ///
    /// Compare two arrays and returns a new array containing the
    /// element - wise maxima
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fmax.html
    ///
    /// @param inArray
    /// @param inScalar
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> fmax(const NdArray<dtype>& inArray, const dtype& inScalar)
    {
        const NdArray<dtype> inArray2 = { inScalar };
        return fmax(inArray, inArray2);
    }

    //============================================================================
    // Method Description:
    /// Element-wise maximum of array elements.
    ///
    /// Compare two arrays and returns a new array containing the
    /// element - wise maxima
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.fmax.html
    ///
    /// @param inScalar
    /// @param inArray
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> fmax(const dtype& inScalar, const NdArray<dtype>& inArray)
    {
        return fmax(inArray, inScalar);
    }
} // namespace nc
