/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.0
///
/// @section License
/// Copyright 2019 David Pilger
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
/// Methods for working with NdArrays
///
#pragma once

#include"NumCpp/Core/Shape.hpp"
#include"NumCpp/Core/Types.hpp"
#include"NumCpp/NdArray/NdArray.hpp"

#include<algorithm>
#include<iostream>
#include<string>
#include<stdexcept>

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
    NdArray<double> average(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE) noexcept
    {
        return inArray.mean(inAxis);
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
        switch (inAxis)
        {
            case Axis::NONE:
            {
                if (inWeights.shape() != inArray.shape())
                {
                    std::string errStr = "ERROR: average: input array and weight values are not consistant.";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                NdArray<double> weightedArray(inArray.shape());
                std::transform(inArray.cbegin(), inArray.cend(), inWeights.cbegin(),
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
                    std::string errStr = "ERROR: average: input array and weights value are not consistant.";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                double weightSum = inWeights.template astype<double>().sum().item();
                NdArray<double> returnArray(1, arrayShape.rows);
                for (uint32 row = 0; row < arrayShape.rows; ++row)
                {
                    NdArray<double> weightedArray(1, arrayShape.cols);
                    std::transform(inArray.cbegin(row), inArray.cend(row), inWeights.cbegin(),
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
                    std::string errStr = "ERROR: average: input array and weight values are not consistant.";
                    std::cerr << errStr << std::endl;
                    throw std::invalid_argument(errStr);
                }

                NdArray<dtype> transposedArray = inArray.transpose();

                const Shape transShape = transposedArray.shape();
                double weightSum = inWeights.template astype<double>().sum().item();
                NdArray<double> returnArray(1, transShape.rows);
                for (uint32 row = 0; row < transShape.rows; ++row)
                {
                    NdArray<double> weightedArray(1, transShape.cols);
                    std::transform(transposedArray.cbegin(row), transposedArray.cend(row), inWeights.cbegin(),
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
}
