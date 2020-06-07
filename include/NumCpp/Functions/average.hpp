/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 2.0.0
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

#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StdComplexOperators.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Internal/TypeTraits.hpp"
#include "NumCpp/Functions/mean.hpp"
#include "NumCpp/NdArray.hpp"

#include <complex>
#include <string>

namespace nc
{
    //============================================================================
    // Method Description:
    ///						Compute the average along the specified axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.average.html
    ///
    /// @param				inArray
    /// @param				inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    auto average(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE) noexcept
    {
        return mean(inArray, inAxis);
    }

    //============================================================================
    // Method Description:
    ///						Compute the weighted average along the specified axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.average.html
    ///
    /// @param				inArray
    /// @param				inWeights
    /// @param  			inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<double> average(const NdArray<dtype>& inArray, const NdArray<dtype>& inWeights, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        switch (inAxis)
        {
            case Axis::NONE:
            {
                if (inWeights.shape() != inArray.shape())
                {
                    THROW_INVALID_ARGUMENT_ERROR("input array and weight values are not consistant.");
                }

                NdArray<double> weightedArray(inArray.shape());
                stl_algorithms::transform(inArray.cbegin(), inArray.cend(), inWeights.cbegin(),
                    weightedArray.begin(), std::multiplies<double>());

                double sum = std::accumulate(weightedArray.begin(), weightedArray.end(), 0.0);
                NdArray<double> returnArray = { sum /= inWeights.template astype<double>().sum().item() };

                return returnArray;
            }
            case Axis::COL:
            {
                const Shape arrayShape = inArray.shape();
                if (inWeights.size() != arrayShape.cols)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input array and weights value are not consistant.");
                }

                double weightSum = inWeights.template astype<double>().sum().item();
                NdArray<double> returnArray(1, arrayShape.rows);
                for (uint32 row = 0; row < arrayShape.rows; ++row)
                {
                    NdArray<double> weightedArray(1, arrayShape.cols);
                    stl_algorithms::transform(inArray.cbegin(row), inArray.cend(row), inWeights.cbegin(),
                        weightedArray.begin(), std::multiplies<double>());

                    double sum = std::accumulate(weightedArray.begin(), weightedArray.end(), 0.0);
                    returnArray(0, row) = sum / weightSum;
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                if (inWeights.size() != inArray.shape().rows)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input array and weight values are not consistant.");
                }

                NdArray<dtype> transposedArray = inArray.transpose();

                const Shape transShape = transposedArray.shape();
                double weightSum = inWeights.template astype<double>().sum().item();
                NdArray<double> returnArray(1, transShape.rows);
                for (uint32 row = 0; row < transShape.rows; ++row)
                {
                    NdArray<double> weightedArray(1, transShape.cols);
                    stl_algorithms::transform(transposedArray.cbegin(row), transposedArray.cend(row), inWeights.cbegin(),
                        weightedArray.begin(), std::multiplies<double>());

                    double sum = std::accumulate(weightedArray.begin(), weightedArray.end(), 0.0);
                    returnArray(0, row) = sum / weightSum;
                }

                return returnArray;
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return NdArray<double>(0);
            }
        }
    }

    //============================================================================
    // Method Description:
    ///						Compute the weighted average along the specified axis.
    ///
    ///                     NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.average.html
    ///
    /// @param				inArray
    /// @param				inWeights
    /// @param  			inAxis (Optional, default NONE)
    /// @return
    ///				NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<double>> average(const NdArray<std::complex<dtype>>& inArray, 
        const NdArray<dtype>& inWeights, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto multiplies = [](const std::complex<dtype>& lhs, dtype rhs) -> std::complex<double>
        {
            return complex_cast<double>(lhs) * static_cast<double>(rhs);
        };

        switch (inAxis)
        {
            case Axis::NONE:
            {
                if (inWeights.shape() != inArray.shape())
                {
                    THROW_INVALID_ARGUMENT_ERROR("input array and weight values are not consistant.");
                }

                NdArray<std::complex<double>> weightedArray(inArray.shape());
                stl_algorithms::transform(inArray.cbegin(), inArray.cend(), inWeights.cbegin(),
                    weightedArray.begin(), multiplies);

                std::complex<double> sum = std::accumulate(weightedArray.begin(), weightedArray.end(), std::complex<double>(0.0));
                NdArray<std::complex<double>> returnArray = { sum /= inWeights.template astype<double>().sum().item() };

                return returnArray;
            }
            case Axis::COL:
            {
                const Shape arrayShape = inArray.shape();
                if (inWeights.size() != arrayShape.cols)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input array and weights value are not consistant.");
                }

                double weightSum = inWeights.template astype<double>().sum().item();
                NdArray<std::complex<double>> returnArray(1, arrayShape.rows);
                for (uint32 row = 0; row < arrayShape.rows; ++row)
                {
                    NdArray<std::complex<double>> weightedArray(1, arrayShape.cols);
                    stl_algorithms::transform(inArray.cbegin(row), inArray.cend(row), inWeights.cbegin(),
                        weightedArray.begin(), multiplies);

                    const std::complex<double> sum = std::accumulate(weightedArray.begin(), weightedArray.end(),
                        std::complex<double>(0.0));
                    returnArray(0, row) = sum / weightSum;
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                if (inWeights.size() != inArray.shape().rows)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input array and weight values are not consistant.");
                }

                NdArray<std::complex<dtype>> transposedArray = inArray.transpose();

                const Shape transShape = transposedArray.shape();
                double weightSum = inWeights.template astype<double>().sum().item();
                NdArray<std::complex<double>> returnArray(1, transShape.rows);
                for (uint32 row = 0; row < transShape.rows; ++row)
                {
                    NdArray<std::complex<double>> weightedArray(1, transShape.cols);
                    stl_algorithms::transform(transposedArray.cbegin(row), transposedArray.cend(row), inWeights.cbegin(),
                        weightedArray.begin(), multiplies);

                    const std::complex<double> sum = std::accumulate(weightedArray.begin(), weightedArray.end(),
                        std::complex<double>(0.0));
                    returnArray(0, row) = sum / weightSum;
                }

                return returnArray;
            }
            default:
            {
                // this isn't actually possible, just putting this here to get rid
                // of the compiler warning.
                return NdArray<std::complex<double>>(0);
            }
        }
    }
}
