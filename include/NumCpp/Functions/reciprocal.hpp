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
#include "NumCpp/Core/Internal/StdComplexOperators.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

#include <complex>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Return the reciprocal of the argument, element-wise.
    ///
    ///						Calculates 1 / x.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.reciprocal.html
    ///
    /// @param
    ///				inArray
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> reciprocal(const NdArray<dtype>& inArray) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        NdArray<double> returnArray(inArray.shape());

        uint32 counter = 0;
        std::for_each(inArray.cbegin(), inArray.cend(),
            [&returnArray, &counter](dtype value) noexcept -> void
            { 
                returnArray[counter++] = 1.0 / static_cast<double>(value);
            });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    ///						Return the reciprocal of the argument, element-wise.
    ///
    ///						Calculates 1 / x.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.reciprocal.html
    ///
    /// @param
    ///				inArray
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<double>> reciprocal(const NdArray<std::complex<dtype>>& inArray) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        NdArray<std::complex<double>> returnArray(inArray.shape());

        uint32 counter = 0;
        std::for_each(inArray.cbegin(), inArray.cend(),
            [&returnArray, &counter](std::complex<dtype> value)  -> void
            { 
                returnArray[counter++] = std::complex<double>(1.0) / complex_cast<double>(value);
            });

        return returnArray;
    }
} // namespace nc
