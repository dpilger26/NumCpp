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

#include <complex>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Returns a std::complex from the input real and imag components
    ///
    /// @param inReal: the real component of the complex number
    /// @return value
    ///
    template<typename dtype>
    auto complex(dtype inReal)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return std::complex<dtype>(inReal);
    }

    //============================================================================
    // Method Description:
    /// Returns a std::complex from the input real and imag components
    ///
    /// @param inReal: the real component of the complex number
    /// @param inImag: the imaginary component of the complex number
    /// @return value
    ///
    template<typename dtype>
    auto complex(dtype inReal, dtype inImag)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return std::complex<dtype>(inReal, inImag);
    }

    //============================================================================
    // Method Description:
    /// Returns a std::complex from the input real and imag components
    ///
    /// @param inReal: the real component of the complex number
    /// @return NdArray
    ///
    template<typename dtype>
    auto complex(const NdArray<dtype>& inReal)
    {
        NdArray<decltype(nc::complex(dtype{ 0 }))> returnArray(inReal.shape());
        stl_algorithms::transform(
            inReal.cbegin(),
            inReal.cend(),
            returnArray.begin(),
            [](dtype real) -> auto{ return nc::complex(real); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Returns a std::complex from the input real and imag components
    ///
    /// @param inReal: the real component of the complex number
    /// @param inImag: the imaginary component of the complex number
    /// @return NdArray
    ///
    template<typename dtype>
    auto complex(const NdArray<dtype>& inReal, const NdArray<dtype>& inImag)
    {
        if (inReal.shape() != inImag.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Input real array must be the same shape as input imag array");
        }

        NdArray<decltype(nc::complex(dtype{ 0 }, dtype{ 0 }))> returnArray(inReal.shape());
        stl_algorithms::transform(
            inReal.cbegin(),
            inReal.cend(),
            inImag.cbegin(),
            returnArray.begin(),
            [](dtype real, dtype imag) -> auto{ return nc::complex(real, imag); });

        return returnArray;
    }
} // namespace nc
