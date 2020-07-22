/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 2.1.0
///
/// @section License
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
/// @section Description
/// Functions for working with NdArrays
///
#pragma once

#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

#include <complex>
#include <numeric>

namespace nc
{
    //===========================================================================
    // Method Description:
    ///						Compute the mean along the specified axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.mean.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> mean(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        switch (inAxis)
        {
            case Axis::NONE:
            {
                auto sum = std::accumulate(inArray.cbegin(), inArray.cend(), 0.0);
                NdArray<double> returnArray = { sum /= static_cast<double>(inArray.size()) };

                return returnArray;
            }
            case Axis::COL:
            {
                NdArray<double> returnArray(1, inArray.numRows());
                for (uint32 row = 0; row < inArray.numRows(); ++row)
                {
                    auto sum = std::accumulate(inArray.cbegin(row), inArray.cend(row), 0.0);
                    returnArray(0, row) = sum / static_cast<double>(inArray.numCols());
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                NdArray<dtype> transposedArray = inArray.transpose();
                NdArray<double> returnArray(1, transposedArray.numRows());
                for (uint32 row = 0; row < transposedArray.numRows(); ++row)
                {
                    auto sum = static_cast<double>(std::accumulate(transposedArray.cbegin(row), transposedArray.cend(row), 0.0));
                    returnArray(0, row) = sum / static_cast<double>(transposedArray.numCols());
                }

                return returnArray;
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
    ///						Compute the mean along the specified axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.mean.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    ///
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<double>> mean(const NdArray<std::complex<dtype>>& inArray, Axis inAxis = Axis::NONE) 
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        switch (inAxis)
        {
            case Axis::NONE:
            {
                auto sum = std::accumulate(inArray.cbegin(), inArray.cend(), std::complex<double>(0.0));
                NdArray<std::complex<double>> returnArray = { sum /= std::complex<double>(inArray.size()) };

                return returnArray;
            }
            case Axis::COL:
            {
                NdArray<std::complex<double>> returnArray(1, inArray.numRows());
                for (uint32 row = 0; row < inArray.numRows(); ++row)
                {
                    auto sum = std::accumulate(inArray.cbegin(row), inArray.cend(row), std::complex<double>(0.0));
                    returnArray(0, row) = sum / std::complex<double>(inArray.numCols());
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                NdArray<std::complex<double>> transposedArray = inArray.transpose();
                NdArray<std::complex<double>> returnArray(1, transposedArray.numRows());
                for (uint32 row = 0; row < transposedArray.numRows(); ++row)
                {
                    auto sum = std::accumulate(transposedArray.cbegin(row), transposedArray.cend(row), 
                        std::complex<double>(0.0));
                    returnArray(0, row) = sum / std::complex<double>(transposedArray.numCols());
                }

                return returnArray;
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
        }
    }
}  // namespace nc
