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

#include <complex>

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc::fft
{
    namespace detail
    {
        //===========================================================================
        // Method Description:
        /// Inverse Fast Fourier Transform
        ///
        /// @param x the data
        /// @param shape Shape (length of each transformed axis) of the output
        ///
        inline NdArray<std::complex<double>> ifft2_internal(const NdArray<std::complex<double>>& x, const Shape& shape)
        {
            if (shape.rows == 0 || shape.cols == 0)
            {
                return {};
            }

            auto result = NdArray<std::complex<double>>(shape.rows, shape.cols);

            stl_algorithms::for_each(result.begin(),
                                     result.end(),
                                     [&](auto& resultElement)
                                     {
                                         const auto i  = &resultElement - result.data();
                                         const auto m  = static_cast<double>(i / shape.cols);
                                         const auto n  = static_cast<double>(i % shape.cols);
                                         resultElement = std::complex<double>{ 0., 0. };
                                         for (auto k = 0u; k < std::min(shape.rows, x.numRows()); ++k)
                                         {
                                             for (auto l = 0u; l < std::min(shape.cols, x.numCols()); ++l)
                                             {
                                                 const auto angle =
                                                     constants::twoPi *
                                                     (((static_cast<double>(k) * m) / static_cast<double>(shape.rows)) +
                                                      ((static_cast<double>(l) * n) / static_cast<double>(shape.cols)));
                                                 resultElement += (x(k, l) * std::polar(1., angle));
                                             }
                                         }
                                         resultElement /= shape.size();
                                     });

            return result;
        }
    } // namespace detail

    //===========================================================================
    // Method Description:
    /// Compute the 2-dimensional inverse discrete Fourier Transform.
    ///
    /// NumPy Reference: <https://numpy.org/doc/2.3/reference/generated/numpy.fft.ifft2.html#numpy.fft.ifft2>
    ///
    /// @param inArray
    /// @param inShape Shape (length of each transformed axis) of the output
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<double>> ifft2(const NdArray<dtype>& inArray, const Shape& inShape)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto data = nc::complex<dtype, double>(inArray);
        return detail::ifft2_internal(data, inShape);
    }

    //===========================================================================
    // Method Description:
    /// Compute the 2-dimensional inverse discrete Fourier Transform.
    ///
    /// NumPy Reference: <https://numpy.org/doc/2.3/reference/generated/numpy.fft.ifft2.html#numpy.fft.ifft2>
    ///
    /// @param inArray
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<double>> ifft2(const NdArray<dtype>& inArray)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return ifft2(inArray, inArray.shape());
    }

    //============================================================================
    // Method Description:
    /// Compute the 2-dimensional inverse discrete Fourier Transform.
    ///
    /// NumPy Reference: <https://numpy.org/doc/2.3/reference/generated/numpy.fft.ifft2.html#numpy.fft.ifft2>
    ///
    /// @param inArray
    /// @param inShape Shape (length of each transformed axis) of the output
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<double>> ifft2(const NdArray<std::complex<dtype>>& inArray, const Shape& inShape)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto data = nc::complex<dtype, double>(inArray);
        return detail::ifft2_internal(data, inShape);
    }

    //============================================================================
    // Method Description:
    /// Compute the 2-dimensional inverse discrete Fourier Transform.
    ///
    /// NumPy Reference: <https://numpy.org/doc/2.3/reference/generated/numpy.fft.ifft2.html#numpy.fft.ifft2>
    ///
    /// @param inArray
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<double>> ifft2(const NdArray<std::complex<dtype>>& inArray)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return ifft2(inArray, inArray.shape());
    }
} // namespace nc::fft
