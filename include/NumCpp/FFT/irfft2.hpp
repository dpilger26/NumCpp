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

#include "NumCpp/Core/Constants.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/FFT/ifft2.hpp"
#include "NumCpp/Functions/complex.hpp"
#include "NumCpp/Functions/real.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc::fft
{
    namespace detail
    {
        //===========================================================================
        // Method Description:
        /// 2D Inverse Fast Fourier Transform for real inputs
        ///
        /// @param x the data
        /// @param shape Shape (length of each transformed axis) of the output
        ///
        inline NdArray<double> irfft2_internal(const NdArray<std::complex<double>>& x, const Shape& shape)
        {
            if (x.size() == 0 || shape.rows == 0 || shape.cols == 0)
            {
                return {};
            }

            const auto necessaryInputPoints = shape.cols / 2 + 1;
            auto       input                = NdArray<std::complex<double>>{};
            if (x.numCols() > necessaryInputPoints)
            {
                input = x(x.rSlice(), Slice(necessaryInputPoints + 1));
            }
            else if (x.numCols() < necessaryInputPoints)
            {
                input = NdArray<std::complex<double>>(shape.rows, necessaryInputPoints).zeros();
                input.put(x.rSlice(), x.cSlice(), x);
            }
            else
            {
                input = x;
            }

            auto realN = 2 * (input.numCols() - 1);
            realN += shape.cols % 2 == 1 ? 1 : 0;
            auto fullOutput = NdArray<std::complex<double>>(shape.rows, realN).zeros();
            for (auto row = 0u; row < input.numRows(); ++row)
            {
                stl_algorithms::copy(input.begin(row), input.end(row), fullOutput.begin(row));
            }
            stl_algorithms::transform(fullOutput.begin(0) + 1,
                                      fullOutput.begin(0) + input.numCols(),
                                      fullOutput.rbegin(0),
                                      [](const auto& value) { return std::conj(value); });
            for (auto col = 1u; col < input.numCols(); ++col)
            {
                stl_algorithms::transform(input.colbegin(col) + 1,
                                          input.colend(col),
                                          fullOutput.rcolbegin(fullOutput.numCols() - col),
                                          [](const auto& value) { return std::conj(value); });
            }

            return real(ifft2_internal(fullOutput, shape));
        }
    } // namespace detail

    //============================================================================
    // Method Description:
    /// Computes the inverse of rfft2.
    ///
    /// NumPy Reference: <https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft2.html>
    ///
    /// @param inArray
    /// @param inShape Shape (length of each transformed axis) of the output
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<double> irfft2(const NdArray<std::complex<dtype>>& inArray, const Shape& inShape)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto data = nc::complex<dtype, double>(inArray);
        return detail::irfft2_internal(data, inShape);
    }

    //============================================================================
    // Method Description:
    /// Computes the inverse of rfft2.
    ///
    /// NumPy Reference: <https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft2.html>
    ///
    /// @param inArray
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<double> irfft2(const NdArray<std::complex<dtype>>& inArray)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto& shape   = inArray.shape();
        const auto  newCols = 2 * (shape.cols - 1);
        return irfft2(inArray, { shape.rows, newCols });
    }
} // namespace nc::fft
