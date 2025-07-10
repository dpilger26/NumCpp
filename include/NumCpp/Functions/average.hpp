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
#include <string>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Internal/StaticAsserts.hpp"
#include "NumCpp/Core/Internal/StdComplexOperators.hpp"
#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Internal/TypeTraits.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/mean.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Compute the average along the specified axis.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.average.html
    ///
    /// @param inArray
    /// @param inAxis (Optional, default NONE)
    /// @return NdArray
    ///
    template<typename dtype>
    auto average(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return mean(inArray, inAxis);
    }

    //============================================================================
    // Method Description:
    /// Compute the weighted average along the specified axis.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.average.html
    ///
    /// @param inArray
    /// @param inWeights
    /// @param inAxis (Optional, default NONE)
    /// @return NdArray
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
                stl_algorithms::transform(inArray.cbegin(),
                                          inArray.cend(),
                                          inWeights.cbegin(),
                                          weightedArray.begin(),
                                          std::multiplies<double>()); // NOLINT(modernize-use-transparent-functors)

                double          sum         = std::accumulate(weightedArray.begin(), weightedArray.end(), 0.);
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

                double          weightSum = inWeights.template astype<double>().sum().item();
                NdArray<double> returnArray(1, arrayShape.rows);
                for (uint32 row = 0; row < arrayShape.rows; ++row)
                {
                    NdArray<double> weightedArray(1, arrayShape.cols);
                    stl_algorithms::transform(inArray.cbegin(row),
                                              inArray.cend(row),
                                              inWeights.cbegin(),
                                              weightedArray.begin(),
                                              std::multiplies<double>()); // NOLINT(modernize-use-transparent-functors)

                    double sum          = std::accumulate(weightedArray.begin(), weightedArray.end(), 0.);
                    returnArray(0, row) = sum / weightSum;
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return average(inArray.transpose(), inWeights, Axis::COL);
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
    /// Compute the weighted average along the specified axis.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.average.html
    ///
    /// @param inArray
    /// @param inWeights
    /// @param inAxis (Optional, default NONE)
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<std::complex<double>>
        average(const NdArray<std::complex<dtype>>& inArray, const NdArray<dtype>& inWeights, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto multiplies = [](const std::complex<dtype>& lhs, dtype rhs) -> std::complex<double>
        { return complex_cast<double>(lhs) * static_cast<double>(rhs); };

        switch (inAxis)
        {
            case Axis::NONE:
            {
                if (inWeights.shape() != inArray.shape())
                {
                    THROW_INVALID_ARGUMENT_ERROR("input array and weight values are not consistant.");
                }

                NdArray<std::complex<double>> weightedArray(inArray.shape());
                stl_algorithms::transform(inArray.cbegin(),
                                          inArray.cend(),
                                          inWeights.cbegin(),
                                          weightedArray.begin(),
                                          multiplies);

                std::complex<double> sum =
                    std::accumulate(weightedArray.begin(), weightedArray.end(), std::complex<double>(0.));
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

                double                        weightSum = inWeights.template astype<double>().sum().item();
                NdArray<std::complex<double>> returnArray(1, arrayShape.rows);
                for (uint32 row = 0; row < arrayShape.rows; ++row)
                {
                    NdArray<std::complex<double>> weightedArray(1, arrayShape.cols);
                    stl_algorithms::transform(inArray.cbegin(row),
                                              inArray.cend(row),
                                              inWeights.cbegin(),
                                              weightedArray.begin(),
                                              multiplies);

                    const std::complex<double> sum =
                        std::accumulate(weightedArray.begin(), weightedArray.end(), std::complex<double>(0.));
                    returnArray(0, row) = sum / weightSum;
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return average(inArray.transpose(), inWeights, Axis::COL);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
        }
    }
} // namespace nc
