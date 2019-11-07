/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.1
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
/// Special Functions
///
#pragma once

#include "NumCpp/NdArray.hpp"
#include "NumCpp/Core/StlAlgorithms.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Functions/exp.hpp"

namespace nc
{
    namespace special
    {
        //============================================================================
        // Method Description:
        ///	The softmax function transforms each element of a collection by computing 
        /// the exponential of each element divided by the sum of the exponentials of all
        /// the elements. That is, if x is a one-dimensional numpy array:
        /// softmax(x) = np.exp(x)/sum(np.exp(x))
        ///
        /// @param      inArray
        /// @param      inAxis (Optional, default NONE)
        /// @return     NdArray<double>
        ///
        template<typename T>
        NdArray<double> softmax(const NdArray<T>& inArray, Axis inAxis = Axis::NONE) noexcept
        {
            switch (inAxis)
            {
                case Axis::NONE:
                {
                    auto returnArray = exp(inArray);
                    returnArray /= returnArray.sum().item();
                    return returnArray;
                }
                case Axis::COL:
                {
                    auto returnArray = exp(inArray);
                    auto expSums = returnArray.sum(inAxis);

                    for (uint32 row = 0; row < returnArray.shape().rows; ++row)
                    {
                        double rowExpSum = expSums[row];
                        stl_algorithms::for_each(returnArray.begin(row), returnArray.end(row), 
                            [rowExpSum](double& value) { value /= rowExpSum; });
                    }

                    return returnArray;
                }
                case Axis::ROW:
                {
                    auto returnArray = exp(inArray).transpose();
                    auto expSums = returnArray.sum(inAxis);

                    for (uint32 row = 0; row < returnArray.shape().rows; ++row)
                    {
                        double rowExpSum = expSums[row];
                        stl_algorithms::for_each(returnArray.begin(row), returnArray.end(row), 
                            [rowExpSum](double& value) { value /= rowExpSum; });
                    }

                    return returnArray.transpose();
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
}
