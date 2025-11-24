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
#include "NumCpp/Functions/complex.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc::fft
{
    namespace detail
    {
        //===========================================================================
        // Method Description:
        /// Fast Fourier Transform
        ///
        /// @param x the data
        /// @param n Length of the transformed axis of the output.
        ///
        inline NdArray<std::complex<double>> internal_rfft(const NdArray<std::complex<double>>& x, uint32 n)
        {
            if (n == 0)
            {
                return {};
            }

            const auto realN  = n / 2 + 1;
            auto       result = NdArray<std::complex<double>>(1, realN);

            stl_algorithms::for_each(result.begin(),
                                     result.end(),
                                     [n, &x, &result](auto& resultElement)
                                     {
                                         const auto k = static_cast<double>(&resultElement - result.data());
                                         const auto minusTwoPiKOverN = -constants::twoPi * k / static_cast<double>(n);
                                         resultElement               = std::complex<double>{ 0., 0. };
                                         for (auto m = 0u; m < std::min(n, x.size()); ++m)
                                         {
                                             const auto angle = minusTwoPiKOverN * static_cast<double>(m);
                                             resultElement += (x[m] * std::polar(1., angle));
                                         }
                                     });

            return result;
        }
    } // namespace detail

    //===========================================================================
    // Method Description:
    /// Compute the one-dimensional discrete Fourier Transform for real input
    ///
    /// NumPy Reference: <https://numpy.org/doc/2.3/reference/generated/numpy.fft.rfft.html#numpy.fft.rfft>
    ///
    /// @param inArray
    /// @param n Length of the transformed axis of the output.
    /// @param inAxis (Optional, default NONE)
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<double>> rfft(const NdArray<dtype>& inArray, uint32 inN, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        switch (inAxis)
        {
            case Axis::NONE:
            {
                const auto data = nc::complex<dtype, double>(inArray);
                return detail::internal_rfft(data, inN);
            }
            case Axis::COL:
            {
                auto        data           = nc::complex<dtype, double>(inArray);
                const auto& shape          = inArray.shape();
                const auto  realN          = inN / 2 + 1;
                auto        result         = NdArray<std::complex<double>>(shape.rows, realN);
                const auto  dataColSlice   = data.cSlice();
                const auto  resultColSlice = result.cSlice();

                for (uint32 row = 0; row < data.numRows(); ++row)
                {
                    const auto rowData   = data(row, dataColSlice);
                    const auto rowResult = detail::internal_rfft(rowData, inN);
                    result.put(row, resultColSlice, rowResult);
                }

                return result;
            }
            case Axis::ROW:
            {
                return rfft(inArray.transpose(), inN, Axis::COL).transpose();
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {};
            }
        }
    }

    //===========================================================================
    // Method Description:
    /// Compute the one-dimensional discrete Fourier Transform for real input
    ///
    /// NumPy Reference: <https://numpy.org/doc/2.3/reference/generated/numpy.fft.rfft.html#numpy.fft.rfft>
    ///
    /// @param inArray
    /// @param inAxis (Optional, default NONE)
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<double>> rfft(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        switch (inAxis)
        {
            case Axis::NONE:
            {
                return rfft(inArray, inArray.size(), inAxis);
            }
            case Axis::COL:
            {
                return rfft(inArray, inArray.numCols(), inAxis);
            }
            case Axis::ROW:
            {
                return rfft(inArray, inArray.numRows(), inAxis);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {};
            }
        }
    }
} // namespace nc::fft
