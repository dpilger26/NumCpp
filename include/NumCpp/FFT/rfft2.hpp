/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2026 David Pilger
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
#include "NumCpp/Functions/complex.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc::fft
{
    namespace detail
    {
        //===========================================================================
        // Method Description:
        /// 2D Fast Fourier Transform for real inputs
        ///
        /// @param x the data
        /// @param shape Shape (length of each transformed axis) of the output
        ///
        inline NdArray<std::complex<double>> rfft2_internal(const NdArray<std::complex<double>>& x, const Shape& shape)
        {
            if (shape.rows == 0 || shape.cols == 0)
            {
                return {};
            }

            const auto realN  = shape.cols / 2 + 1;
            auto       result = NdArray<std::complex<double>>(shape.rows, realN);

            stl_algorithms::for_each(result.begin(),
                                     result.end(),
                                     [&](auto& resultElement)
                                     {
                                         const auto i  = &resultElement - result.data();
                                         const auto k  = static_cast<double>(i / realN);
                                         const auto l  = static_cast<double>(i % realN);
                                         resultElement = std::complex<double>{ 0., 0. };
                                         for (auto m = 0u; m < std::min(shape.rows, x.numRows()); ++m)
                                         {
                                             for (auto n = 0u; n < std::min(shape.cols, x.numCols()); ++n)
                                             {
                                                 const auto angle =
                                                     -constants::twoPi *
                                                     (((static_cast<double>(m) * k) / static_cast<double>(shape.rows)) +
                                                      ((static_cast<double>(n) * l) / static_cast<double>(shape.cols)));
                                                 resultElement += (x(m, n) * std::polar(1., angle));
                                             }
                                         }
                                     });

            return result;
        }
    } // namespace detail

    //============================================================================
    // Method Description:
    /// Compute the 2-dimensional FFT of a real array.
    ///
    /// NumPy Reference: <https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft2.html>
    ///
    /// @param inArray
    /// @param inShape Shape (length of each transformed axis) of the output
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<double>> rfft2(const NdArray<dtype>& inArray, const Shape& inShape)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto data = nc::complex<dtype, double>(inArray);
        return detail::rfft2_internal(data, inShape);
    }

    //============================================================================
    // Method Description:
    /// Compute the 2-dimensional FFT of a real array.
    ///
    /// NumPy Reference: <https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft2.html>
    ///
    /// @param inArray
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<double>> rfft2(const NdArray<dtype>& inArray)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return rfft2(inArray, inArray.shape());
    }
} // namespace nc::fft
