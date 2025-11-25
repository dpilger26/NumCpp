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

#include "NumCpp/Core/Constants.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/FFT/ifft.hpp"
#include "NumCpp/Functions/complex.hpp"
#include "NumCpp/Functions/real.hpp"
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
        /// @param n Length of the transformed axis of the output.
        ///
        inline NdArray<double> irfft_internal(const NdArray<std::complex<double>>& x, uint32 n)
        {
            if (x.size() == 0 || n == 0)
            {
                return {};
            }

            const auto necessaryInputPoints = n / 2 + 1;
            auto       input                = NdArray<std::complex<double>>{};
            if (x.size() > necessaryInputPoints)
            {
                input = x.flatten()(0, Slice(necessaryInputPoints + 1));
            }
            else if (x.size() < necessaryInputPoints)
            {
                input = NdArray<std::complex<double>>(1, necessaryInputPoints).zeros();
                stl_algorithms::copy(x.begin(), x.end(), input.begin());
            }
            else
            {
                input = x;
            }

            auto realN = 2 * (input.size() - 1);
            realN += n % 2 == 1 ? 1 : 0;
            auto fullOutput = NdArray<std::complex<double>>(1, realN);
            stl_algorithms::copy(input.begin(), input.end(), fullOutput.begin());
            stl_algorithms::transform(fullOutput.begin() + 1,
                                      fullOutput.begin() + input.size(),
                                      fullOutput.rbegin(),
                                      [](const auto& value) { return std::conj(value); });
            return real(ifft_internal(fullOutput, n));
        }
    } // namespace detail

    //============================================================================
    // Method Description:
    /// Compute the one-dimensional inverse discrete Fourier Transform for real inputs.
    ///
    /// NumPy Reference: <https://numpy.org/doc/2.3/reference/generated/numpy.fft.irfft.html#numpy.fft.irfft>
    ///
    /// @param inArray
    /// @param n Length of the transformed axis of the output.
    /// @param inAxis (Optional, default NONE)
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<double> irfft(const NdArray<std::complex<dtype>>& inArray, uint32 inN, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        switch (inAxis)
        {
            case Axis::NONE:
            {
                const auto data = nc::complex<dtype, double>(inArray);
                return detail::irfft_internal(data, inN);
            }
            case Axis::COL:
            {
                const auto  data           = nc::complex<dtype, double>(inArray);
                const auto& shape          = inArray.shape();
                auto        result         = NdArray<double>(shape.rows, inN);
                const auto  dataColSlice   = data.cSlice();
                const auto  resultColSlice = result.cSlice();

                for (uint32 row = 0; row < data.numRows(); ++row)
                {
                    const auto rowData   = data(row, dataColSlice);
                    const auto rowResult = detail::irfft_internal(rowData, inN);
                    result.put(row, resultColSlice, rowResult);
                }

                return result;
            }
            case Axis::ROW:
            {
                return irfft(inArray.transpose(), inN, Axis::COL).transpose();
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {};
            }
        }
    }

    //============================================================================
    // Method Description:
    /// Compute the one-dimensional inverse discrete Fourier Transform for real inputs.
    ///
    /// NumPy Reference: <https://numpy.org/doc/2.3/reference/generated/numpy.fft.irfft.html#numpy.fft.irfft>
    ///
    /// @param inArray
    /// @param inAxis (Optional, default NONE)
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<double> irfft(const NdArray<std::complex<dtype>>& inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        switch (inAxis)
        {
            case Axis::NONE:
            {
                return irfft(inArray, 2 * (inArray.size() - 1), inAxis);
            }
            case Axis::COL:
            {
                return irfft(inArray, 2 * (inArray.numCols() - 1), inAxis);
            }
            case Axis::ROW:
            {
                return irfft(inArray, 2 * (inArray.numRows() - 1), inAxis);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {};
            }
        }
    }
} // namespace nc::fft
